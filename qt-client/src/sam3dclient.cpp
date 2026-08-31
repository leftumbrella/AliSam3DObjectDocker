#include "sam3dclient.h"

#include <QBuffer>
#include <QHttpMultiPart>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>

namespace {

void appendField(QHttpMultiPart *multipart, const QByteArray &name, const QByteArray &value)
{
    QHttpPart part;
    part.setHeader(QNetworkRequest::ContentDispositionHeader,
                   QStringLiteral("form-data; name=\"%1\"")
                       .arg(QString::fromLatin1(name)));
    part.setBody(value);
    multipart->append(part);
}

void appendPng(QHttpMultiPart *multipart,
               const QByteArray &name,
               const QByteArray &filename,
               const QByteArray &content)
{
    QHttpPart part;
    part.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("image/png"));
    part.setHeader(QNetworkRequest::ContentDispositionHeader,
                   QStringLiteral("form-data; name=\"%1\"; filename=\"%2\"")
                       .arg(QString::fromLatin1(name), QString::fromLatin1(filename)));
    part.setBody(content);
    multipart->append(part);
}

int responseStatus(QNetworkReply *reply)
{
    return reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
}

bool succeeded(QNetworkReply *reply, int statusCode)
{
    return reply->error() == QNetworkReply::NoError
           && statusCode >= 200
           && statusCode < 300;
}

} // namespace

Sam3dClient::Sam3dClient(QObject *parent)
    : QObject(parent),
      m_network(new QNetworkAccessManager(this)),
      m_endpoint(defaultEndpoint())
{
}

QUrl Sam3dClient::defaultEndpoint()
{
    return QUrl(QStringLiteral("https://samd-object-duanxppffx.cn-shenzhen.fcapp.run"));
}

bool Sam3dClient::setEndpoint(const QUrl &endpoint, QString *error)
{
    QUrl normalized = endpoint;
    normalized.setQuery(QString());
    normalized.setFragment(QString());

    QString path = normalized.path();
    while (path.endsWith(QLatin1Char('/')) && path.size() > 1)
        path.chop(1);
    if (path == QStringLiteral("/"))
        path.clear();
    normalized.setPath(path);

    const QString scheme = normalized.scheme().toLower();
    if ((scheme != QStringLiteral("http") && scheme != QStringLiteral("https"))
        || normalized.host().isEmpty()) {
        if (error)
            *error = QStringLiteral("函数地址必须是完整的 http:// 或 https:// URL");
        return false;
    }

    m_endpoint = normalized;
    return true;
}

QUrl Sam3dClient::route(const QString &path) const
{
    QUrl url = m_endpoint;
    QString basePath = url.path();
    if (basePath.endsWith(QLatin1Char('/')))
        basePath.chop(1);
    url.setPath(basePath + (path.startsWith(QLatin1Char('/')) ? path
                                                              : QStringLiteral("/") + path));
    return url;
}

QNetworkRequest Sam3dClient::makeRequest(const QString &path,
                                         const QByteArray &accept,
                                         int timeoutMs) const
{
    QNetworkRequest request(route(path));
    request.setRawHeader("Accept", accept);
    request.setRawHeader("User-Agent", "SAM3DQtClient/1.0");
    request.setRawHeader("Cache-Control", "no-store");
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    request.setTransferTimeout(timeoutMs);
#else
    Q_UNUSED(timeoutMs)
#endif
    return request;
}

QByteArray Sam3dClient::encodePng(const QImage &image, QString *error)
{
    if (image.isNull()) {
        if (error)
            *error = QStringLiteral("没有可上传的图像");
        return {};
    }

    QByteArray bytes;
    QBuffer buffer(&bytes);
    if (!buffer.open(QIODevice::WriteOnly) || !image.save(&buffer, "PNG")) {
        if (error)
            *error = QStringLiteral("无法将图像编码为 PNG");
        return {};
    }
    return bytes;
}

QString Sam3dClient::errorMessage(QNetworkReply *reply,
                                  const QByteArray &body,
                                  int statusCode)
{
    const QJsonDocument document = QJsonDocument::fromJson(body);
    if (document.isObject()) {
        const QJsonValue detail = document.object().value(QStringLiteral("detail"));
        if (detail.isString() && !detail.toString().trimmed().isEmpty())
            return detail.toString().trimmed();
        if (detail.isArray()) {
            QStringList messages;
            for (const QJsonValue &entry : detail.toArray()) {
                if (entry.isObject()) {
                    const QString message = entry.toObject().value(QStringLiteral("msg")).toString();
                    if (!message.isEmpty())
                        messages.append(message);
                }
            }
            if (!messages.isEmpty())
                return messages.join(QStringLiteral("；"));
        }
    }

    if (statusCode > 0)
        return QStringLiteral("函数服务返回 HTTP %1").arg(statusCode);
    const QString networkError = reply->errorString().trimmed();
    return networkError.isEmpty() ? QStringLiteral("无法连接函数服务") : networkError;
}

void Sam3dClient::checkReady()
{
    if (m_readyReply) {
        QNetworkReply *previous = m_readyReply;
        m_readyReply = nullptr;
        previous->abort();
    }

    QNetworkReply *reply = m_network->get(
        makeRequest(QStringLiteral("/readyz"), "application/json", 30000));
    m_readyReply = reply;
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        const QByteArray body = reply->readAll();
        const int statusCode = responseStatus(reply);
        const bool current = m_readyReply == reply;
        if (current)
            m_readyReply = nullptr;

        if (reply->error() == QNetworkReply::OperationCanceledError || !current) {
            reply->deleteLater();
            return;
        }

        if (!succeeded(reply, statusCode)) {
            emit readinessFinished(false, errorMessage(reply, body, statusCode));
            reply->deleteLater();
            return;
        }

        const QJsonDocument document = QJsonDocument::fromJson(body);
        if (!document.isObject()) {
            emit readinessFinished(false, QStringLiteral("/readyz 返回了无效 JSON"));
            reply->deleteLater();
            return;
        }
        const QJsonObject payload = document.object();
        const bool ready = payload.value(QStringLiteral("ready")).toBool(false);
        emit readinessFinished(ready,
                               ready ? QStringLiteral("SAM3 与 SAM3D 已完成预热")
                                     : QStringLiteral("模型仍在预热，请稍后重试"));
        reply->deleteLater();
    });
}

void Sam3dClient::segment(const QImage &image,
                          const QVector<Point> &points,
                          quint64 revision)
{
    QString encodingError;
    const QByteArray imageBytes = encodePng(image, &encodingError);
    if (imageBytes.isEmpty()) {
        emit requestFailed(Operation::Segmentation, encodingError, 0, revision);
        return;
    }

    QJsonArray pointArray;
    for (const Point &point : points) {
        QJsonObject item;
        item.insert(QStringLiteral("x"), point.x);
        item.insert(QStringLiteral("y"), point.y);
        item.insert(QStringLiteral("label"), point.label == 0 ? 0 : 1);
        pointArray.append(item);
    }
    if (pointArray.isEmpty()) {
        emit requestFailed(Operation::Segmentation, QStringLiteral("至少需要一个点选"), 0, revision);
        return;
    }

    if (m_segmentReply) {
        QNetworkReply *previous = m_segmentReply;
        m_segmentReply = nullptr;
        previous->abort();
    }

    auto *multipart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
    appendPng(multipart, "image", "input.png", imageBytes);
    appendField(multipart, "points", QJsonDocument(pointArray).toJson(QJsonDocument::Compact));

    QNetworkReply *reply = m_network->post(
        makeRequest(QStringLiteral("/segment"), "image/png", 120000), multipart);
    multipart->setParent(reply);
    m_segmentReply = reply;
    emit segmentationBusyChanged(true);

    connect(reply, &QNetworkReply::finished, this, [this, reply, revision] {
        const QByteArray body = reply->readAll();
        const int statusCode = responseStatus(reply);
        const bool current = m_segmentReply == reply;
        if (current) {
            m_segmentReply = nullptr;
            emit segmentationBusyChanged(false);
        }

        if (reply->error() == QNetworkReply::OperationCanceledError || !current) {
            reply->deleteLater();
            return;
        }
        if (!succeeded(reply, statusCode)) {
            emit requestFailed(Operation::Segmentation,
                               errorMessage(reply, body, statusCode),
                               statusCode,
                               revision);
            reply->deleteLater();
            return;
        }

        const QImage mask = QImage::fromData(body);
        if (mask.isNull()) {
            emit requestFailed(Operation::Segmentation,
                               QStringLiteral("/segment 未返回有效的 PNG Mask"),
                               statusCode,
                               revision);
            reply->deleteLater();
            return;
        }
        emit segmentationFinished(mask,
                                  QString::fromLatin1(reply->rawHeader("X-Segment-Score")),
                                  revision);
        reply->deleteLater();
    });
}

void Sam3dClient::generate(const QImage &image, const QImage &mask, int seed)
{
    QString encodingError;
    const QByteArray imageBytes = encodePng(image, &encodingError);
    if (imageBytes.isEmpty()) {
        emit requestFailed(Operation::Generation, encodingError, 0, 0);
        return;
    }
    const QByteArray maskBytes = encodePng(mask, &encodingError);
    if (maskBytes.isEmpty()) {
        emit requestFailed(Operation::Generation, encodingError, 0, 0);
        return;
    }

    cancelGeneration();
    auto *multipart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
    appendPng(multipart, "image", "input.png", imageBytes);
    appendPng(multipart, "mask", "mask.png", maskBytes);
    appendField(multipart, "seed", QByteArray::number(seed));

    QNetworkReply *reply = m_network->post(
        makeRequest(QStringLiteral("/generate"), "model/gltf-binary", 15 * 60 * 1000),
        multipart);
    multipart->setParent(reply);
    m_generationReply = reply;
    emit generationBusyChanged(true);

    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        const QByteArray body = reply->readAll();
        const int statusCode = responseStatus(reply);
        const bool current = m_generationReply == reply;
        if (current) {
            m_generationReply = nullptr;
            emit generationBusyChanged(false);
        }

        if (reply->error() == QNetworkReply::OperationCanceledError || !current) {
            reply->deleteLater();
            return;
        }
        if (!succeeded(reply, statusCode)) {
            emit requestFailed(Operation::Generation,
                               errorMessage(reply, body, statusCode),
                               statusCode,
                               0);
            reply->deleteLater();
            return;
        }
        if (body.size() < 12 || body.left(4) != QByteArray("glTF", 4)) {
            emit requestFailed(Operation::Generation,
                               QStringLiteral("/generate 未返回有效的 GLB 2.0 文件"),
                               statusCode,
                               0);
            reply->deleteLater();
            return;
        }
        emit generationFinished(body);
        reply->deleteLater();
    });
}

void Sam3dClient::cancelSegmentation()
{
    if (!m_segmentReply)
        return;
    QNetworkReply *reply = m_segmentReply;
    m_segmentReply = nullptr;
    reply->abort();
    emit segmentationBusyChanged(false);
}

void Sam3dClient::cancelGeneration()
{
    if (!m_generationReply)
        return;
    QNetworkReply *reply = m_generationReply;
    m_generationReply = nullptr;
    reply->abort();
    emit generationBusyChanged(false);
}
