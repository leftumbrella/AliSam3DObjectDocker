#pragma once

#include "modeldata.h"

#include <QImage>
#include <QOpenGLFunctions_2_1>
#include <QOpenGLWidget>
#include <QPointF>
#include <QTimer>
#include <QVector>

class QKeyEvent;
class QMouseEvent;
class QPainter;
class QWheelEvent;

class EditorCanvas : public QOpenGLWidget, protected QOpenGLFunctions_2_1
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
    UiState uiState() const { return m_state; }

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void leaveEvent(QEvent *event) override;

private:
    struct SelectionMark {
        QPointF center;
        bool positive = true;
        qreal radiusX = 86.0;
        qreal radiusY = 58.0;
    };

    enum class ToolIcon {
        ArrowLeft,
        Close,
        Save,
        Image,
        Rotate,
        Cube,
        Cursor,
        Lasso,
        Brush,
        Eraser,
        Sparkles,
        Eye,
        Grid,
        Plus,
        Minus,
        Undo,
        Trash,
        Download,
        Fullscreen,
        Home,
        Check,
        Warning
    };

    QPointF toDesign(const QPointF &point) const;
    QRectF designRect(qreal x, qreal y, qreal width, qreal height) const;
    bool isResultState() const;

    void renderModel();
    void drawScene(QPainter &painter);
    void drawImageCover(QPainter &painter, const QRectF &target);
    void drawSelections(QPainter &painter, const QRectF &clipRect);
    void drawTopBar(QPainter &painter);
    void drawBottomEditor(QPainter &painter);
    void drawResultOverlay(QPainter &painter);
    void drawModal(QPainter &painter);
    void drawSavedToast(QPainter &painter);
    void drawIcon(QPainter &painter, ToolIcon icon, const QPointF &center, const QColor &color, qreal size = 20.0);
    void drawToolButton(QPainter &painter,
                        const QRectF &rect,
                        ToolIcon icon,
                        const QString &tooltip,
                        bool active = false,
                        bool enabled = true);
    void drawButton(QPainter &painter,
                    const QRectF &rect,
                    const QString &text,
                    bool primary,
                    ToolIcon icon = ToolIcon::Cursor);
    void drawSpinner(QPainter &painter, const QPointF &center, qreal radius);

    void addSelection(const QPointF &position, bool positive);
    void openImage();
    void rotateImage();
    void openModel();
    void saveModel();
    void beginGeneration();
    void completeGeneration();
    void resetModelView();
    void toggleFullscreen();
    void showTransientMessage(const QString &message);
    void handleClick(const QPointF &position);
    QString tooltipAt(const QPointF &position) const;

    QImage m_sourceImage;
    QImage m_templateSelected;
    QImage m_templateConfirm;
    QImage m_templateGenerating;
    QImage m_templateFailed;
    QImage m_templateSaved;
    QVector<SelectionMark> m_marks;
    ModelData m_model;
    UiState m_state = UiState::Waiting;
    bool m_addMode = true;
    bool m_usingBundledSample = true;
    bool m_captureEnabled = true;
    bool m_savedToastVisible = false;
    bool m_demoStateLocked = false;
    int m_activeTool = 0;
    int m_loadingPhase = 0;
    QString m_imageName = QStringLiteral("叶片表皮");
    QString m_modelName = QStringLiteral("SAM 有机体模型");
    QString m_transientMessage;

    float m_rotationX = -14.0f;
    float m_rotationY = 26.0f;
    float m_zoom = 0.82f;
    QPointF m_pan;
    QPointF m_lastMouse;
    QPointF m_pressDesign;
    bool m_mouseMoved = false;
    bool m_rotatingModel = false;
    bool m_panningModel = false;
    bool m_draggingWindow = false;
    QPoint m_windowDragOffset;

    QTimer m_generationTimer;
    QTimer m_spinnerTimer;
    QTimer m_toastTimer;
    QTimer m_messageTimer;
    GLuint m_overlayTexture = 0;
};
