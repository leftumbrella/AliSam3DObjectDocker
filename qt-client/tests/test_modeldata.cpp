#include "editorcanvas.h"
#include "modeldata.h"

#include <QBuffer>
#include <QDataStream>
#include <QFile>
#include <QHash>
#include <QImage>
#include <QPainter>
#include <QPushButton>
#include <QStackedWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTimer>
#include <QtTest>

namespace {

QByteArray minimalGlb()
{
    QByteArray binary;
    QDataStream binaryStream(&binary, QIODevice::WriteOnly);
    binaryStream.setByteOrder(QDataStream::LittleEndian);
    binaryStream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    binaryStream << float(-1.0f) << float(-1.0f) << float(0.0f);
    binaryStream << float(1.0f) << float(-1.0f) << float(0.0f);
    binaryStream << float(0.0f) << float(1.0f) << float(0.0f);
    binaryStream << quint16(0) << quint16(1) << quint16(2);

    const int bufferLength = binary.size();
    while (binary.size() % 4)
        binary.append('\0');

    QByteArray json = QByteArrayLiteral(
        "{\"asset\":{\"version\":\"2.0\"},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"indices\":1}]}],"
        "\"buffers\":[{\"byteLength\":BUFFER_LENGTH}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":6}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
        "{\"bufferView\":1,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}]}"
    );
    json.replace("BUFFER_LENGTH", QByteArray::number(bufferLength));
    while (json.size() % 4)
        json.append(' ');

    const quint32 totalLength = quint32(12 + 8 + json.size() + 8 + binary.size());
    QByteArray glb;
    QDataStream stream(&glb, QIODevice::WriteOnly);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream << quint32(0x46546c67) << quint32(2) << totalLength;
    stream << quint32(json.size()) << quint32(0x4e4f534a);
    stream.writeRawData(json.constData(), json.size());
    stream << quint32(binary.size()) << quint32(0x004e4942);
    stream.writeRawData(binary.constData(), binary.size());
    return glb;
}

QByteArray pngMask(const QSize &size, const QRect &selected = QRect())
{
    QImage mask(size, QImage::Format_Grayscale8);
    if (selected.isNull()) {
        mask.fill(255);
    } else {
        mask.fill(0);
        QPainter painter(&mask);
        painter.fillRect(selected, Qt::white);
    }
    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    mask.save(&buffer, "PNG");
    return bytes;
}

class FakeFunctionServer : public QObject
{
public:
    FakeFunctionServer(const QSize &imageSize, const QByteArray &glb, QObject *parent = nullptr)
        : QObject(parent), m_mask(pngMask(imageSize)), m_glb(glb)
    {
        connect(&m_server, &QTcpServer::newConnection, this, [this] {
            while (m_server.hasPendingConnections()) {
                QTcpSocket *socket = m_server.nextPendingConnection();
                m_buffers.insert(socket, QByteArray());
                connect(socket, &QTcpSocket::readyRead, this, [this, socket] {
                    consume(socket);
                });
                connect(socket, &QTcpSocket::disconnected, this, [this, socket] {
                    m_buffers.remove(socket);
                    socket->deleteLater();
                });
                consume(socket);
            }
        });
    }

    bool listen()
    {
        return m_server.listen(QHostAddress::LocalHost, 0);
    }

    QUrl endpoint() const
    {
        return QUrl(QStringLiteral("http://127.0.0.1:%1").arg(m_server.serverPort()));
    }

    QVector<QByteArray> segmentBodies;
    QVector<QByteArray> generationBodies;

    void setMask(const QByteArray &mask) { m_mask = mask; }
    void setSegmentDelay(int delayMs) { m_segmentDelayMs = delayMs; }

private:
    void consume(QTcpSocket *socket)
    {
        QByteArray &buffer = m_buffers[socket];
        buffer.append(socket->readAll());
        const int headerEnd = buffer.indexOf("\r\n\r\n");
        if (headerEnd < 0)
            return;

        const QList<QByteArray> headerLines = buffer.left(headerEnd).split('\n');
        int contentLength = 0;
        for (const QByteArray &rawLine : headerLines) {
            const QByteArray line = rawLine.trimmed();
            if (line.toLower().startsWith("content-length:"))
                contentLength = line.mid(line.indexOf(':') + 1).trimmed().toInt();
        }
        const int requestLength = headerEnd + 4 + contentLength;
        if (buffer.size() < requestLength)
            return;

        const QByteArray request = buffer.left(requestLength);
        const QByteArray requestLine = headerLines.isEmpty() ? QByteArray() : headerLines.first().trimmed();
        const QByteArray body = request.mid(headerEnd + 4, contentLength);
        if (requestLine.startsWith("GET /readyz ")) {
            reply(socket, QByteArrayLiteral("{\"ready\":true}"), "application/json");
        } else if (requestLine.startsWith("POST /segment ")) {
            segmentBodies.append(body);
            const auto sendMask = [this, socket] {
                reply(socket, m_mask, "image/png",
                      QByteArrayLiteral("X-Segment-Score: 0.960000\r\n"));
            };
            if (m_segmentDelayMs > 0)
                QTimer::singleShot(m_segmentDelayMs, this, sendMask);
            else
                sendMask();
        } else if (requestLine.startsWith("POST /generate ")) {
            generationBodies.append(body);
            reply(socket, m_glb, "model/gltf-binary");
        } else {
            reply(socket, QByteArrayLiteral("{\"detail\":\"not found\"}"),
                  "application/json", QByteArray(), 404, "Not Found");
        }
        buffer.remove(0, requestLength);
    }

    static void reply(QTcpSocket *socket,
                      const QByteArray &body,
                      const QByteArray &contentType,
                      const QByteArray &extraHeaders = QByteArray(),
                      int status = 200,
                      const QByteArray &statusText = QByteArrayLiteral("OK"))
    {
        QByteArray response = "HTTP/1.1 " + QByteArray::number(status) + " " + statusText + "\r\n";
        response += "Content-Type: " + contentType + "\r\n";
        response += "Content-Length: " + QByteArray::number(body.size()) + "\r\n";
        response += extraHeaders;
        response += "Connection: close\r\n\r\n";
        response += body;
        socket->write(response);
        socket->disconnectFromHost();
    }

    QHash<QTcpSocket *, QByteArray> m_buffers;
    QTcpServer m_server;
    QByteArray m_mask;
    QByteArray m_glb;
    int m_segmentDelayMs = 0;
};

QPushButton *buttonWithText(QWidget &parent, const QString &text)
{
    for (QPushButton *button : parent.findChildren<QPushButton *>()) {
        if (button->text().contains(text))
            return button;
    }
    return nullptr;
}

QColor colorAtWidgetPoint(const QPixmap &pixmap,
                          const QSize &widgetSize,
                          const QPoint &widgetPoint)
{
    const QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB32);
    const int x = qBound(0,
                         qRound(widgetPoint.x() * image.width()
                                / qreal(widgetSize.width())),
                         image.width() - 1);
    const int y = qBound(0,
                         qRound(widgetPoint.y() * image.height()
                                / qreal(widgetSize.height())),
                         image.height() - 1);
    return image.pixelColor(x, y);
}

int blendChannel(int foreground, int alpha, int background)
{
    return qRound((foreground * alpha + background * (255 - alpha)) / 255.0);
}

int baseChannelBeforeTint(int displayed, int tint, int tintAlpha)
{
    return qBound(0,
                  qRound((displayed * 255.0 - tint * tintAlpha)
                         / (255 - tintAlpha)),
                  255);
}

} // namespace

class ModelDataTest : public QObject
{
    Q_OBJECT

private slots:
    void proceduralModelHasGeometry();
    void glbLoadsAndRoundTrips();
    void invalidModelIsRejected();
    void editorUsesNativeControls();
    void editorRendersFunctionMaskWithCorrectAlpha();
    void editorQueuesLatestPointsWhileSegmentationIsBusy();
    void editorUsesFunctionComputeForSelectionAndGeneration();
};

void ModelDataTest::proceduralModelHasGeometry()
{
    ModelData model;
    model.createOrganicSample();

    QVERIFY(!model.isEmpty());
    QVERIFY(model.vertices.size() > 2000);
    QVERIFY(model.triangleCount() > 2000);
    QCOMPARE(model.vertices.size(), model.normals.size());
    QCOMPARE(model.vertices.size(), model.colors.size());
}

void ModelDataTest::glbLoadsAndRoundTrips()
{
    const QByteArray source = minimalGlb();
    ModelData model;
    QString error;
    QVERIFY2(model.loadGlbData(source, &error), qPrintable(error));
    QCOMPARE(model.vertices.size(), 3);
    QCOMPARE(model.triangleCount(), 1);
    QCOMPARE(model.colors.size(), model.vertices.size());

    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString output = directory.filePath(QStringLiteral("triangle.glb"));
    QVERIFY2(model.saveGlb(output, &error), qPrintable(error));
    QFile file(output);
    QVERIFY(file.open(QIODevice::ReadOnly));
    QCOMPARE(file.readAll(), source);

    ModelData reloaded;
    QVERIFY2(reloaded.load(output, &error), qPrintable(error));
    QCOMPARE(reloaded.triangleCount(), 1);
}

void ModelDataTest::invalidModelIsRejected()
{
    ModelData model;
    QString error;
    QVERIFY(!model.loadGlbData(QByteArrayLiteral("not a model"), &error));
    QVERIFY(!error.isEmpty());
    QVERIFY(model.isEmpty());
}

void ModelDataTest::editorUsesNativeControls()
{
    EditorCanvas editor;
    editor.resize(1280, 800);
    editor.setDemoState(QStringLiteral("waiting"));
    editor.show();
    QVERIFY(QTest::qWaitForWindowExposed(&editor, 1200));

    QVERIFY2(editor.findChildren<QPushButton *>().size() >= 8,
             "The editor must use native interactive Qt controls.");
    QVERIFY2(editor.findChild<QStackedWidget *>(QStringLiteral("StateModal")),
             "Generation states must use a real modal stack.");
    editor.close();
}

void ModelDataTest::editorRendersFunctionMaskWithCorrectAlpha()
{
    const QImage source(QStringLiteral(":/design/sample-microbe.png"));
    QVERIFY(!source.isNull());
    FakeFunctionServer server(source.size(), minimalGlb());
    server.setMask(pngMask(source.size(), QRect(650, 250, 500, 500)));
    QVERIFY(server.listen());

    EditorCanvas editor;
    QString endpointError;
    QVERIFY2(editor.setServiceEndpoint(server.endpoint(), &endpointError), qPrintable(endpointError));
    editor.resize(1280, 800);
    editor.show();
    QVERIFY(QTest::qWaitForWindowExposed(&editor, 1200));

    QWidget *imageView = editor.findChild<QWidget *>(QStringLiteral("ImageSelectionView"));
    QVERIFY(imageView);
    const QPixmap before = editor.grab();
    QTest::mouseClick(imageView, Qt::LeftButton, Qt::NoModifier, QPoint(360, 470));
    QPushButton *generateButton = buttonWithText(editor, QStringLiteral("生成 3D 模型"));
    QVERIFY(generateButton);
    QTRY_VERIFY_WITH_TIMEOUT(generateButton->isEnabled(), 3000);
    const QPixmap after = editor.grab();

    const QPoint samplePoint(726, 338);
    const QColor beforeColor = colorAtWidgetPoint(before, editor.size(), samplePoint);
    const QColor afterColor = colorAtWidgetPoint(after, editor.size(), samplePoint);
    const int maskAlpha = qRound(255 * 0.62);
    const int tintAlpha = 22;
    const int tintChannels[] = {4, 9, 18};
    const int maskChannels[] = {235, 46, 55};
    const int beforeChannels[] = {beforeColor.red(), beforeColor.green(), beforeColor.blue()};
    const int afterChannels[] = {afterColor.red(), afterColor.green(), afterColor.blue()};
    for (int channel = 0; channel < 3; ++channel) {
        const int base = baseChannelBeforeTint(beforeChannels[channel],
                                               tintChannels[channel],
                                               tintAlpha);
        const int masked = blendChannel(maskChannels[channel], maskAlpha, base);
        const int expected = blendChannel(tintChannels[channel], tintAlpha, masked);
        QVERIFY2(qAbs(afterChannels[channel] - expected) <= 5,
                 qPrintable(QStringLiteral("Mask channel %1 was %2, expected %3")
                                .arg(channel)
                                .arg(afterChannels[channel])
                                .arg(expected)));
    }
    editor.close();
}

void ModelDataTest::editorQueuesLatestPointsWhileSegmentationIsBusy()
{
    const QImage source(QStringLiteral(":/design/sample-microbe.png"));
    QVERIFY(!source.isNull());
    FakeFunctionServer server(source.size(), minimalGlb());
    server.setSegmentDelay(350);
    QVERIFY(server.listen());

    EditorCanvas editor;
    QString endpointError;
    QVERIFY2(editor.setServiceEndpoint(server.endpoint(), &endpointError), qPrintable(endpointError));
    editor.resize(1280, 800);
    editor.show();
    QVERIFY(QTest::qWaitForWindowExposed(&editor, 1200));

    QWidget *imageView = editor.findChild<QWidget *>(QStringLiteral("ImageSelectionView"));
    QVERIFY(imageView);
    QTest::mouseClick(imageView, Qt::LeftButton, Qt::NoModifier, QPoint(500, 340));
    QTRY_COMPARE_WITH_TIMEOUT(server.segmentBodies.size(), 1, 2000);
    QTest::mouseClick(imageView, Qt::LeftButton, Qt::NoModifier, QPoint(650, 400));
    QTRY_COMPARE_WITH_TIMEOUT(server.segmentBodies.size(), 2, 3000);
    QCOMPARE(server.segmentBodies.first().count("\"label\""), 1);
    QCOMPARE(server.segmentBodies.last().count("\"label\""), 2);
    editor.close();
}

void ModelDataTest::editorUsesFunctionComputeForSelectionAndGeneration()
{
    const QImage source(QStringLiteral(":/design/sample-microbe.png"));
    QVERIFY(!source.isNull());
    FakeFunctionServer server(source.size(), minimalGlb());
    QVERIFY(server.listen());

    EditorCanvas editor;
    QString endpointError;
    QVERIFY2(editor.setServiceEndpoint(server.endpoint(), &endpointError), qPrintable(endpointError));
    editor.resize(1280, 800);
    editor.show();
    QVERIFY(QTest::qWaitForWindowExposed(&editor, 1200));

    QWidget *imageView = editor.findChild<QWidget *>(QStringLiteral("ImageSelectionView"));
    QVERIFY(imageView);
    QTest::mouseClick(imageView, Qt::LeftButton, Qt::NoModifier, QPoint(500, 340));
    QTRY_VERIFY_WITH_TIMEOUT(server.segmentBodies.size() >= 1, 3000);
    QVERIFY(server.segmentBodies.first().contains("name=\"points\""));
    QVERIFY(server.segmentBodies.first().contains("\"label\":1"));

    QPushButton *generateButton = buttonWithText(editor, QStringLiteral("生成 3D 模型"));
    QVERIFY(generateButton);
    QTRY_VERIFY_WITH_TIMEOUT(generateButton->isEnabled(), 3000);

    QPushButton *subtractButton = buttonWithText(editor, QStringLiteral("减少选区"));
    QVERIFY(subtractButton);
    QTest::mouseClick(subtractButton, Qt::LeftButton);
    QTest::mouseClick(imageView, Qt::LeftButton, Qt::NoModifier, QPoint(650, 400));
    QTRY_VERIFY_WITH_TIMEOUT(server.segmentBodies.size() >= 2, 3000);
    QVERIFY(server.segmentBodies.last().contains("\"label\":1"));
    QVERIFY(server.segmentBodies.last().contains("\"label\":0"));
    QTRY_VERIFY_WITH_TIMEOUT(generateButton->isEnabled(), 3000);

    QTest::mouseClick(generateButton, Qt::LeftButton);
    QCOMPARE(int(editor.uiState()), int(EditorCanvas::UiState::CreditConfirm));
    QPushButton *confirmButton = buttonWithText(editor, QStringLiteral("确认转换"));
    QVERIFY(confirmButton);
    QTest::mouseClick(confirmButton, Qt::LeftButton);
    QCOMPARE(int(editor.uiState()), int(EditorCanvas::UiState::Generating));
    QTRY_COMPARE_WITH_TIMEOUT(int(editor.uiState()), int(EditorCanvas::UiState::Result), 5000);
    QCOMPARE(server.generationBodies.size(), 1);
    QVERIFY(server.generationBodies.first().contains("name=\"image\""));
    QVERIFY(server.generationBodies.first().contains("name=\"mask\""));
    QVERIFY(server.generationBodies.first().contains("name=\"seed\""));
    QVERIFY(!server.generationBodies.first().contains("output_format"));
    editor.close();
}

QTEST_MAIN(ModelDataTest)

#include "test_modeldata.moc"
