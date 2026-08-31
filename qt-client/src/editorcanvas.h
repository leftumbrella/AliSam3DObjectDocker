#pragma once

#include "modeldata.h"
#include "sam3dclient.h"

#include <QIcon>
#include <QImage>
#include <QTimer>
#include <QVector>
#include <QWidget>

class ImageSelectionView;
class ModelViewport;
class ToggleSwitch;
class QFrame;
class QGraphicsOpacityEffect;
class QKeyEvent;
class QLabel;
class QProgressBar;
class QPropertyAnimation;
class QPushButton;
class QResizeEvent;
class QStackedWidget;
class QToolButton;
class QUrl;

class EditorCanvas : public QWidget
{
public:
    enum class UiState {
        Waiting,
        Selected,
        CreditConfirm,
        Generating,
        Failed,
        Result
    };

    explicit EditorCanvas(QWidget *parent = nullptr);
    ~EditorCanvas() override;

    void setDemoState(const QString &stateName);
    bool setServiceEndpoint(const QUrl &endpoint, QString *error = nullptr);
    QUrl serviceEndpoint() const;
    UiState uiState() const { return m_state; }

protected:
    void resizeEvent(QResizeEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    void buildInterface();
    void buildTopBar();
    void buildStatusBar();
    void buildToolBars();
    void buildModal();
    void buildToast();
    void layoutInterface();
    void applyState(bool animateModal = true);
    void setState(UiState state, bool animateModal = true);
    void updateSelectionState();

    QPushButton *createTextButton(const QString &text,
                                  const QString &objectName,
                                  QWidget *parent,
                                  const QIcon &icon = QIcon());
    QToolButton *createToolButton(const QString &tooltip,
                                  const QIcon &icon,
                                  QWidget *parent,
                                  bool checkable = false);

    void openImage();
    void rotateImage();
    void openModel();
    void configureService();
    void saveModel();
    void requestSegmentation();
    void dispatchSegmentation();
    void beginGeneration();
    void completeGeneration(const QByteArray &glb);
    void toggleFullscreen();
    void showToast(const QString &title = QStringLiteral("已保存为模型"),
                   const QString &detail = QString(),
                   bool showAction = true);

    QImage m_sourceImage;
    QImage m_maskImage;
    ModelData m_model;
    Sam3dClient *m_client = nullptr;
    UiState m_state = UiState::Waiting;
    QString m_imageName = QStringLiteral("叶片表皮");
    QString m_modelName = QStringLiteral("SAM 有机体模型");
    bool m_addMode = true;
    bool m_savedToastVisible = false;
    bool m_demoStateLocked = false;
    bool m_segmentBusy = false;
    bool m_segmentQueued = false;
    bool m_maskReady = false;
    bool m_serviceReady = false;
    quint64 m_selectionRevision = 0;
    QString m_selectionError;
    QString m_serviceDetail;

    QWidget *m_contentLayer = nullptr;
    ImageSelectionView *m_imageView = nullptr;
    ModelViewport *m_modelView = nullptr;

    QFrame *m_topBar = nullptr;
    QPushButton *m_backButton = nullptr;
    QLabel *m_titleLabel = nullptr;
    QPushButton *m_exitButton = nullptr;
    QPushButton *m_saveButton = nullptr;

    QFrame *m_statusBar = nullptr;
    QLabel *m_statusIcon = nullptr;
    QLabel *m_statusLabel = nullptr;
    QPushButton *m_addButton = nullptr;
    QPushButton *m_subtractButton = nullptr;
    QPushButton *m_generateButton = nullptr;

    QFrame *m_leftTools = nullptr;
    QFrame *m_centerTools = nullptr;
    QFrame *m_aiTools = nullptr;
    QVector<QToolButton *> m_leftToolButtons;
    QVector<QToolButton *> m_centerToolButtons;
    QLabel *m_aiLabel = nullptr;
    ToggleSwitch *m_aiSwitch = nullptr;

    QPushButton *m_downloadButton = nullptr;
    QPushButton *m_fullscreenButton = nullptr;

    QWidget *m_modalShade = nullptr;
    QStackedWidget *m_modalStack = nullptr;
    QGraphicsOpacityEffect *m_modalOpacity = nullptr;
    QPropertyAnimation *m_modalAnimation = nullptr;
    QProgressBar *m_generationProgress = nullptr;
    QLabel *m_failedBody = nullptr;

    QFrame *m_toast = nullptr;
    QLabel *m_toastTitle = nullptr;
    QLabel *m_toastDetail = nullptr;
    QPushButton *m_toastAction = nullptr;

    QTimer m_toastTimer;
};
