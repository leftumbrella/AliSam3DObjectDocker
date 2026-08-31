#include "editorcanvas.h"

#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QOpenGLContext>
#include <QScreen>
#include <QSurfaceFormat>
#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    QApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
    QGuiApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    QSurfaceFormat format;
    format.setVersion(2, 1);
    format.setProfile(QSurfaceFormat::CompatibilityProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(4);
    format.setSwapInterval(1);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication application(argc, argv);
    application.setApplicationName(QStringLiteral("SAM3DQtClient"));
    application.setOrganizationName(QStringLiteral("SAM 3D"));

    QString stateName;
    QString endpoint;
    QString screenshotPath;
    bool maximize = false;
    const QStringList arguments = application.arguments();
    for (const QString &argument : arguments) {
        if (argument.startsWith(QStringLiteral("--state=")))
            stateName = argument.mid(QStringLiteral("--state=").size());
        else if (argument.startsWith(QStringLiteral("--endpoint=")))
            endpoint = argument.mid(QStringLiteral("--endpoint=").size());
        else if (argument.startsWith(QStringLiteral("--screenshot=")))
            screenshotPath = argument.mid(QStringLiteral("--screenshot=").size());
        else if (argument == QStringLiteral("--maximized"))
            maximize = true;
    }

    EditorCanvas window;
    window.resize(1280, 800);
    if (!endpoint.isEmpty()) {
        QString error;
        if (!window.setServiceEndpoint(QUrl(endpoint), &error))
            qWarning() << error;
    }
    if (!stateName.isEmpty())
        window.setDemoState(stateName);

    if (maximize) {
        window.showMaximized();
    } else {
        if (QScreen *screen = QGuiApplication::primaryScreen()) {
            const QRect available = screen->availableGeometry();
            window.move(available.center() - QPoint(window.width() / 2, window.height() / 2));
        }
        window.show();
    }

    if (!screenshotPath.isEmpty()) {
        QTimer::singleShot(900, &window, [&window, screenshotPath] {
            QFileInfo outputInfo(screenshotPath);
            QDir().mkpath(outputInfo.absolutePath());
            const QImage image = window.grab().toImage();
            const bool saved = image.save(outputInfo.absoluteFilePath());
            QCoreApplication::exit(saved ? 0 : 2);
        });
    }

    return application.exec();
}
