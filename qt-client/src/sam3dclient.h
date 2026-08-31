#pragma once

#include <QImage>
#include <QObject>
#include <QPointer>
#include <QUrl>
#include <QVector>

class QNetworkAccessManager;
class QNetworkReply;
class QNetworkRequest;

class Sam3dClient : public QObject
{
    Q_OBJECT

public:
    struct Point {
        int x = 0;
        int y = 0;
        int label = 1;
    };

    enum class Operation {
        Readiness,
        Segmentation,
        Generation
    };
    Q_ENUM(Operation)

    explicit Sam3dClient(QObject *parent = nullptr);

    static QUrl defaultEndpoint();
    bool setEndpoint(const QUrl &endpoint, QString *error = nullptr);
    QUrl endpoint() const { return m_endpoint; }

    void checkReady();
    void segment(const QImage &image, const QVector<Point> &points, quint64 revision);
    void generate(const QImage &image, const QImage &mask, int seed = 42);
    void cancelSegmentation();
    void cancelGeneration();

signals:
    void readinessFinished(bool ready, const QString &detail);
    void segmentationBusyChanged(bool busy);
    void segmentationFinished(const QImage &mask,
                              const QString &score,
                              quint64 revision);
    void generationBusyChanged(bool busy);
    void generationFinished(const QByteArray &glb);
    void requestFailed(Sam3dClient::Operation operation,
                       const QString &message,
                       int statusCode,
                       quint64 revision);

private:
    QUrl route(const QString &path) const;
    QNetworkRequest makeRequest(const QString &path,
                                const QByteArray &accept,
                                int timeoutMs) const;
    static QByteArray encodePng(const QImage &image, QString *error);
    static QString errorMessage(QNetworkReply *reply,
                                const QByteArray &body,
                                int statusCode);

    QNetworkAccessManager *m_network = nullptr;
    QUrl m_endpoint;
    QPointer<QNetworkReply> m_readyReply;
    QPointer<QNetworkReply> m_segmentReply;
    QPointer<QNetworkReply> m_generationReply;
};

Q_DECLARE_METATYPE(Sam3dClient::Operation)
