#include "editorcanvas.h"
#include "modeldata.h"

#include <QDataStream>
#include <QFile>
#include <QTemporaryDir>
#include <QtTest>

class ModelDataTest : public QObject
{
    Q_OBJECT

private slots:
    void proceduralModelHasGeometry();
    void asciiPlyRoundTrip();
    void binaryLittleEndianPlyLoads();
    void objLoadsAndTriangulates();
    void editorGenerationFlow();
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

void ModelDataTest::asciiPlyRoundTrip()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());

    ModelData source;
    source.createOrganicSample();
    const QString fileName = directory.filePath(QStringLiteral("sample.ply"));
    QString error;
    QVERIFY2(source.savePly(fileName, &error), qPrintable(error));

    ModelData loaded;
    QVERIFY2(loaded.load(fileName, &error), qPrintable(error));
    QCOMPARE(loaded.vertices.size(), source.vertices.size());
    QCOMPARE(loaded.triangleCount(), source.triangleCount());
    QCOMPARE(loaded.colors.size(), loaded.vertices.size());
}

void ModelDataTest::binaryLittleEndianPlyLoads()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString fileName = directory.filePath(QStringLiteral("triangle-binary.ply"));
    QFile file(fileName);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("ply\nformat binary_little_endian 1.0\n");
    file.write("element vertex 3\nproperty float x\nproperty float y\nproperty float z\n");
    file.write("element face 1\nproperty list uchar int vertex_indices\nend_header\n");
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    stream << float(0.0f) << float(0.0f) << float(0.0f);
    stream << float(1.0f) << float(0.0f) << float(0.0f);
    stream << float(0.0f) << float(1.0f) << float(0.0f);
    stream << quint8(3) << qint32(0) << qint32(1) << qint32(2);
    file.close();

    ModelData model;
    QString error;
    QVERIFY2(model.load(fileName, &error), qPrintable(error));
    QCOMPARE(model.vertices.size(), 3);
    QCOMPARE(model.triangleCount(), 1);
}

void ModelDataTest::objLoadsAndTriangulates()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString fileName = directory.filePath(QStringLiteral("quad.obj"));
    QFile file(fileName);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write("v -1 -1 0 0.1 0.8 0.4\n");
    file.write("v  1 -1 0 0.1 0.8 0.4\n");
    file.write("v  1  1 0 0.1 0.8 0.4\n");
    file.write("v -1  1 0 0.1 0.8 0.4\n");
    file.write("f 1 2 3 4\n");
    file.close();

    ModelData model;
    QString error;
    QVERIFY2(model.load(fileName, &error), qPrintable(error));
    QCOMPARE(model.vertices.size(), 4);
    QCOMPARE(model.triangleCount(), 2);
}

void ModelDataTest::editorGenerationFlow()
{
    EditorCanvas editor;
    editor.resize(1280, 800);
    editor.setDemoState(QStringLiteral("waiting"));
    editor.show();
    QVERIFY(QTest::qWaitForWindowExposed(&editor, 1200));

    QTest::mouseClick(&editor, Qt::LeftButton, Qt::NoModifier, QPoint(500, 340));
    QCOMPARE(int(editor.uiState()), int(EditorCanvas::UiState::Selected));

    QTest::mouseClick(&editor, Qt::LeftButton, Qt::NoModifier, QPoint(1030, 666));
    QCOMPARE(int(editor.uiState()), int(EditorCanvas::UiState::CreditConfirm));

    QTest::mouseClick(&editor, Qt::LeftButton, Qt::NoModifier, QPoint(785, 496));
    QCOMPARE(int(editor.uiState()), int(EditorCanvas::UiState::Generating));
    QTRY_COMPARE_WITH_TIMEOUT(int(editor.uiState()), int(EditorCanvas::UiState::Result), 2600);
    editor.close();
}

QTEST_MAIN(ModelDataTest)

#include "test_modeldata.moc"
