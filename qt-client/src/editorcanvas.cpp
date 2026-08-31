#include "editorcanvas.h"

#include <QAbstractButton>
#include <QApplication>
#include <QButtonGroup>
#include <QEasingCurve>
#include <QFileDialog>
#include <QFileInfo>
#include <QGraphicsOpacityEffect>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QOpenGLFunctions_2_1>
#include <QOpenGLWidget>
#include <QPainter>
#include <QPainterPath>
#include <QProgressBar>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QResizeEvent>
#include <QSettings>
#include <QShortcut>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWindow>
#include <QtMath>

#include <functional>

namespace Theme {

const QColor canvas(7, 13, 25);
const QColor surface(14, 21, 36);
const QColor surfaceRaised(22, 31, 50);
const QColor surfaceSoft(38, 48, 69);
const QColor border(57, 70, 94);
const QColor text(240, 244, 251);
const QColor secondary(171, 181, 200);
const QColor muted(111, 125, 148);
const QColor primary(42, 104, 255);
const QColor mask(235, 58, 67);
const QColor success(53, 203, 142);
const QColor warning(255, 179, 71);

QFont font(int pixelSize, int weight = QFont::Normal)
{
    QFont result(QStringLiteral("Microsoft YaHei UI"));
    result.setPixelSize(pixelSize);
    result.setWeight(weight);
    result.setHintingPreference(QFont::PreferFullHinting);
    return result;
}

} // namespace Theme

namespace {

enum class IconKind {
    Back,
    Save,
    Image,
    Rotate,
    Settings,
    Cube,
    Target,
    Cursor,
    Line,
    Circle,
    Lasso,
    Angle,
    Rectangle,
    Ellipse,
    Text,
    Trash,
    Undo,
    Download,
    Fullscreen
};

QIcon makeIcon(IconKind kind, const QColor &color = Theme::text)
{
    constexpr int logicalSize = 36;
    constexpr qreal ratio = 2.0;
    QPixmap pixmap(logicalSize * int(ratio), logicalSize * int(ratio));
    pixmap.setDevicePixelRatio(ratio);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(color, 1.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.setBrush(Qt::NoBrush);
    painter.translate(logicalSize / 2.0, logicalSize / 2.0);

    switch (kind) {
    case IconKind::Back:
        painter.drawLine(QPointF(6, -8), QPointF(-3, 0));
        painter.drawLine(QPointF(-3, 0), QPointF(6, 8));
        break;
    case IconKind::Save:
        painter.drawRoundedRect(QRectF(-9, -10, 18, 20), 2, 2);
        painter.drawRect(QRectF(-5, -9, 9, 6));
        painter.drawRoundedRect(QRectF(-5, 2, 10, 6), 1, 1);
        break;
    case IconKind::Image:
        painter.drawRoundedRect(QRectF(-10, -9, 20, 18), 3, 3);
        painter.drawEllipse(QPointF(4, -3), 2, 2);
        painter.drawPolyline(QPolygonF() << QPointF(-8, 6) << QPointF(-2, 0)
                                         << QPointF(2, 4) << QPointF(5, 1) << QPointF(9, 6));
        break;
    case IconKind::Rotate:
        painter.drawArc(QRectF(-9, -9, 18, 18), 25 * 16, 285 * 16);
        painter.drawPolyline(QPolygonF() << QPointF(6, -10) << QPointF(10, -7) << QPointF(7, -3));
        break;
    case IconKind::Settings:
        painter.drawLine(QPointF(-10, -6), QPointF(10, -6));
        painter.drawLine(QPointF(-10, 0), QPointF(10, 0));
        painter.drawLine(QPointF(-10, 6), QPointF(10, 6));
        painter.drawEllipse(QPointF(-3, -6), 2, 2);
        painter.drawEllipse(QPointF(4, 0), 2, 2);
        painter.drawEllipse(QPointF(-1, 6), 2, 2);
        break;
    case IconKind::Cube:
        painter.drawPolygon(QPolygonF() << QPointF(0, -10) << QPointF(9, -5) << QPointF(9, 6)
                                        << QPointF(0, 11) << QPointF(-9, 6) << QPointF(-9, -5));
        painter.drawLine(QPointF(0, 0), QPointF(0, 10));
        painter.drawLine(QPointF(0, 0), QPointF(9, -5));
        painter.drawLine(QPointF(0, 0), QPointF(-9, -5));
        break;
    case IconKind::Target:
        painter.drawEllipse(QPointF(0, 0), 9, 9);
        painter.drawEllipse(QPointF(0, 0), 3, 3);
        painter.drawPoint(QPointF(0, 0));
        break;
    case IconKind::Cursor:
        painter.drawPolygon(QPolygonF() << QPointF(-7, -10) << QPointF(8, 2)
                                        << QPointF(1, 4) << QPointF(4, 11)
                                        << QPointF(0, 12) << QPointF(-3, 5) << QPointF(-8, 9));
        break;
    case IconKind::Line:
        painter.drawLine(QPointF(-9, 8), QPointF(9, -8));
        painter.drawEllipse(QPointF(-9, 8), 1.5, 1.5);
        painter.drawEllipse(QPointF(9, -8), 1.5, 1.5);
        break;
    case IconKind::Circle:
        painter.drawEllipse(QPointF(0, 0), 8, 8);
        break;
    case IconKind::Lasso: {
        QPainterPath path;
        path.moveTo(-8, 2);
        path.cubicTo(-10, -8, 2, -11, 8, -5);
        path.cubicTo(13, 1, 5, 8, -2, 8);
        path.cubicTo(-5, 8, -8, 6, -8, 2);
        painter.drawPath(path);
        painter.drawLine(QPointF(-3, 8), QPointF(-6, 11));
        break;
    }
    case IconKind::Angle:
        painter.drawPolyline(QPolygonF() << QPointF(-9, 8) << QPointF(-3, -8) << QPointF(2, 7) << QPointF(10, 7));
        break;
    case IconKind::Rectangle:
        painter.drawRoundedRect(QRectF(-9, -7, 18, 14), 2, 2);
        break;
    case IconKind::Ellipse:
        painter.drawEllipse(QRectF(-10, -6, 20, 12));
        break;
    case IconKind::Text:
        painter.setFont(Theme::font(20, QFont::DemiBold));
        painter.drawText(QRectF(-12, -13, 24, 26), Qt::AlignCenter, QStringLiteral("T"));
        break;
    case IconKind::Trash:
        painter.drawRoundedRect(QRectF(-7, -6, 14, 16), 2, 2);
        painter.drawLine(QPointF(-9, -7), QPointF(9, -7));
        painter.drawLine(QPointF(-3, -10), QPointF(3, -10));
        painter.drawLine(QPointF(-3, -3), QPointF(-3, 6));
        painter.drawLine(QPointF(3, -3), QPointF(3, 6));
        break;
    case IconKind::Undo:
        painter.drawArc(QRectF(-8, -8, 18, 17), -75 * 16, 245 * 16);
        painter.drawPolyline(QPolygonF() << QPointF(-10, -7) << QPointF(-10, 0) << QPointF(-3, -2));
        break;
    case IconKind::Download:
        painter.drawLine(QPointF(0, -10), QPointF(0, 5));
        painter.drawPolyline(QPolygonF() << QPointF(-5, 0) << QPointF(0, 5) << QPointF(5, 0));
        painter.drawLine(QPointF(-9, 10), QPointF(9, 10));
        break;
    case IconKind::Fullscreen:
        painter.drawPolyline(QPolygonF() << QPointF(-2, -10) << QPointF(-10, -10) << QPointF(-10, -2));
        painter.drawPolyline(QPolygonF() << QPointF(2, -10) << QPointF(10, -10) << QPointF(10, -2));
        painter.drawPolyline(QPolygonF() << QPointF(-10, 2) << QPointF(-10, 10) << QPointF(-2, 10));
        painter.drawPolyline(QPolygonF() << QPointF(10, 2) << QPointF(10, 10) << QPointF(2, 10));
        break;
    }

    painter.end();
    QIcon icon;
    icon.addPixmap(pixmap, QIcon::Normal, QIcon::Off);
    return icon;
}

bool isSplitState(EditorCanvas::UiState state)
{
    return state == EditorCanvas::UiState::CreditConfirm
           || state == EditorCanvas::UiState::Generating
           || state == EditorCanvas::UiState::Failed
           || state == EditorCanvas::UiState::Result;
}

class TitleBar : public QFrame
{
public:
    explicit TitleBar(QWidget *parent = nullptr)
        : QFrame(parent)
    {
        setMouseTracking(true);
    }

protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() != Qt::LeftButton) {
            QFrame::mousePressEvent(event);
            return;
        }
        m_dragging = true;
        m_offset = event->globalPos() - window()->frameGeometry().topLeft();
        event->accept();
    }

    void mouseMoveEvent(QMouseEvent *event) override
    {
        if (m_dragging && !window()->isMaximized() && !window()->isFullScreen())
            window()->move(event->globalPos() - m_offset);
        event->accept();
    }

    void mouseReleaseEvent(QMouseEvent *event) override
    {
        m_dragging = false;
        event->accept();
    }

    void mouseDoubleClickEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton) {
            window()->isMaximized() ? window()->showNormal() : window()->showMaximized();
            event->accept();
            return;
        }
        QFrame::mouseDoubleClickEvent(event);
    }

private:
    bool m_dragging = false;
    QPoint m_offset;
};

} // namespace

class ToggleSwitch : public QAbstractButton
{
public:
    explicit ToggleSwitch(QWidget *parent = nullptr)
        : QAbstractButton(parent)
    {
        setCheckable(true);
        setChecked(true);
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::StrongFocus);
        setFixedSize(46, 26);
        setAccessibleName(QStringLiteral("AI 捕捉微生物"));
        setToolTip(QStringLiteral("开启或关闭 AI 捕捉"));
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);

        QColor track = isChecked() ? Theme::primary : QColor(60, 72, 96);
        if (underMouse())
            track = track.lighter(112);
        if (isDown())
            track = track.darker(115);

        const QRectF trackRect(1, 2, width() - 2, height() - 4);
        painter.setPen(QPen(hasFocus() ? QColor(126, 165, 255) : QColor(89, 104, 132), 1));
        painter.setBrush(track);
        painter.drawRoundedRect(trackRect, trackRect.height() / 2, trackRect.height() / 2);

        const qreal diameter = 18;
        const qreal x = isChecked() ? width() - diameter - 4 : 4;
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(247, 250, 255));
        painter.drawEllipse(QRectF(x, 4, diameter, diameter));
    }

    void enterEvent(QEvent *event) override
    {
        update();
        QAbstractButton::enterEvent(event);
    }

    void leaveEvent(QEvent *event) override
    {
        update();
        QAbstractButton::leaveEvent(event);
    }
};

class ImageSelectionView : public QWidget
{
public:
    struct SelectionMark {
        QPointF imagePoint;
        bool positive = true;
    };

    explicit ImageSelectionView(QWidget *parent = nullptr)
        : QWidget(parent)
    {
        setAttribute(Qt::WA_OpaquePaintEvent);
        setMouseTracking(true);
        setFocusPolicy(Qt::StrongFocus);
        setCursor(Qt::CrossCursor);
        setAccessibleName(QStringLiteral("显微图像选区画布"));
    }

    void setSourceImage(const QImage &image)
    {
        m_image = image;
        m_maskOverlay = QImage();
        update();
    }

    void setMask(const QImage &mask)
    {
        if (mask.isNull()) {
            m_maskOverlay = QImage();
            update();
            return;
        }

        const QImage source = mask.convertToFormat(QImage::Format_ARGB32);
        m_maskOverlay = QImage(source.size(), QImage::Format_ARGB32_Premultiplied);
        for (int y = 0; y < source.height(); ++y) {
            const QRgb *input = reinterpret_cast<const QRgb *>(source.constScanLine(y));
            QRgb *output = reinterpret_cast<QRgb *>(m_maskOverlay.scanLine(y));
            for (int x = 0; x < source.width(); ++x) {
                const int gray = qGray(input[x]);
                const int sourceAlpha = qAlpha(input[x]);
                const int coverage = sourceAlpha < 255 ? qMax(gray, sourceAlpha) : gray;
                output[x] = qRgba(235, 46, 55, qRound(coverage * 0.62));
            }
        }
        update();
    }

    void clearMask()
    {
        setMask(QImage());
    }

    void setSplit(bool split)
    {
        if (m_split == split)
            return;
        m_split = split;
        update();
    }

    void setAddMode(bool addMode)
    {
        m_addMode = addMode;
        setCursor(m_interactive ? (addMode ? Qt::CrossCursor : Qt::PointingHandCursor)
                                : Qt::ArrowCursor);
    }

    void setInteractive(bool interactive)
    {
        m_interactive = interactive;
        setCursor(interactive ? (m_addMode ? Qt::CrossCursor : Qt::PointingHandCursor)
                              : Qt::ArrowCursor);
    }

    int selectionCount() const { return m_marks.size(); }

    int positiveCount() const
    {
        int count = 0;
        for (const SelectionMark &mark : m_marks) {
            if (mark.positive)
                ++count;
        }
        return count;
    }

    QVector<Sam3dClient::Point> points() const
    {
        QVector<Sam3dClient::Point> result;
        result.reserve(m_marks.size());
        for (const SelectionMark &mark : m_marks) {
            Sam3dClient::Point point;
            point.x = qBound(0, qRound(mark.imagePoint.x()), qMax(0, m_image.width() - 1));
            point.y = qBound(0, qRound(mark.imagePoint.y()), qMax(0, m_image.height() - 1));
            point.label = mark.positive ? 1 : 0;
            result.append(point);
        }
        return result;
    }

    void clearSelections(bool notify = true)
    {
        m_marks.clear();
        clearMask();
        update();
        if (notify && selectionChanged)
            selectionChanged();
    }

    void removeLastSelection()
    {
        if (m_marks.isEmpty())
            return;
        m_marks.removeLast();
        update();
        if (selectionChanged)
            selectionChanged();
    }

    void setDemoSelections(bool enabled)
    {
        m_marks.clear();
        if (enabled && !m_image.isNull()) {
            m_marks.append({QPointF(m_image.width() * 0.275,
                                          m_image.height() * 0.31), true});
            m_marks.append({QPointF(m_image.width() * 0.565,
                                          m_image.height() * 0.53), true});
        }
        update();
    }

    std::function<void()> selectionChanged;

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
        painter.fillRect(rect(), Theme::canvas);

        const qreal fullWidth = targetWidth();
        if (!m_image.isNull()) {
            const QRectF target(0, 0, fullWidth, height());
            const QRectF source = imageSourceRect();
            painter.drawImage(target, m_image, source);
            if (!m_maskOverlay.isNull())
                painter.drawImage(target, m_maskOverlay, source);
        } else {
            QLinearGradient gradient(0, 0, 0, height());
            gradient.setColorAt(0, QColor(49, 61, 72));
            gradient.setColorAt(1, QColor(15, 24, 31));
            painter.fillRect(rect(), gradient);
        }

        painter.fillRect(rect(), QColor(4, 9, 18, 22));

        for (const SelectionMark &mark : m_marks) {
            const QPointF center = imageToTarget(mark.imagePoint);
            const QColor accent = mark.positive ? Theme::primary : Theme::mask;
            painter.setPen(QPen(QColor(247, 250, 255), 2.0));
            painter.setBrush(accent);
            painter.drawEllipse(center, 7.0, 7.0);
            painter.setPen(QPen(QColor(255, 255, 255), 1.6, Qt::SolidLine, Qt::RoundCap));
            painter.drawLine(center + QPointF(-3.0, 0.0), center + QPointF(3.0, 0.0));
            if (mark.positive)
                painter.drawLine(center + QPointF(0.0, -3.0), center + QPointF(0.0, 3.0));
        }
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        if (m_interactive
            && (event->button() == Qt::LeftButton || event->button() == Qt::RightButton)) {
            m_pressPosition = event->localPos();
            m_pressButton = event->button();
            m_pressed = true;
            event->accept();
            return;
        }
        QWidget::mousePressEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent *event) override
    {
        if (!m_pressed || event->button() != m_pressButton) {
            QWidget::mouseReleaseEvent(event);
            return;
        }
        m_pressed = false;
        if ((event->localPos() - m_pressPosition).manhattanLength() > 6) {
            event->accept();
            return;
        }

        const QPointF imagePoint = targetToImage(event->localPos());
        if (imagePoint.x() < 0.0 || imagePoint.y() < 0.0)
            return;
        SelectionMark mark;
        mark.imagePoint = imagePoint;
        mark.positive = m_pressButton == Qt::RightButton ? false : m_addMode;
        m_marks.append(mark);
        update();
        if (selectionChanged)
            selectionChanged();
        event->accept();
    }

private:
    qreal targetWidth() const
    {
        return m_split ? width() * 2.0 : width();
    }

    QRectF imageSourceRect() const
    {
        if (m_image.isNull() || height() <= 0 || targetWidth() <= 0)
            return {};
        const qreal imageAspect = qreal(m_image.width()) / qreal(m_image.height());
        const qreal targetAspect = targetWidth() / qreal(height());
        QRectF source(0, 0, m_image.width(), m_image.height());
        if (imageAspect > targetAspect) {
            const qreal sourceWidth = m_image.height() * targetAspect;
            source.setLeft((m_image.width() - sourceWidth) * 0.5);
            source.setWidth(sourceWidth);
        } else {
            const qreal sourceHeight = m_image.width() / targetAspect;
            source.setTop((m_image.height() - sourceHeight) * 0.5);
            source.setHeight(sourceHeight);
        }
        return source;
    }

    QPointF targetToImage(const QPointF &point) const
    {
        const QRectF source = imageSourceRect();
        if (source.isEmpty() || point.x() < 0.0 || point.x() > targetWidth()
            || point.y() < 0.0 || point.y() > height())
            return QPointF(-1.0, -1.0);
        return QPointF(source.left() + point.x() / targetWidth() * source.width(),
                       source.top() + point.y() / qreal(height()) * source.height());
    }

    QPointF imageToTarget(const QPointF &point) const
    {
        const QRectF source = imageSourceRect();
        if (source.isEmpty())
            return {};
        return QPointF((point.x() - source.left()) / source.width() * targetWidth(),
                       (point.y() - source.top()) / source.height() * height());
    }

    QImage m_image;
    QImage m_maskOverlay;
    QVector<SelectionMark> m_marks;
    bool m_split = false;
    bool m_addMode = true;
    bool m_interactive = true;
    bool m_pressed = false;
    Qt::MouseButton m_pressButton = Qt::NoButton;
    QPointF m_pressPosition;
};

class ModelViewport : public QOpenGLWidget, protected QOpenGLFunctions_2_1
{
public:
    explicit ModelViewport(ModelData *model, QWidget *parent = nullptr)
        : QOpenGLWidget(parent), m_model(model)
    {
        setAttribute(Qt::WA_OpaquePaintEvent);
        setFocusPolicy(Qt::StrongFocus);
        setMouseTracking(true);
        setCursor(Qt::OpenHandCursor);
        setAccessibleName(QStringLiteral("3D 模型交互预览"));
        setToolTip(QStringLiteral("左键旋转，右键平移，滚轮缩放，双击复位"));
    }

    void resetView()
    {
        m_rotationX = -14.0f;
        m_rotationY = 26.0f;
        m_zoom = 0.82f;
        m_pan = QPointF();
        update();
    }

    void rotateBy(float x, float y)
    {
        m_rotationX = qBound(-89.0f, m_rotationX + x, 89.0f);
        m_rotationY += y;
        update();
    }

    void zoomBy(float factor)
    {
        m_zoom = qBound(0.35f, m_zoom * factor, 4.5f);
        update();
    }

protected:
    void initializeGL() override
    {
        initializeOpenGLFunctions();
        glClearColor(0.04f, 0.075f, 0.10f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_MULTISAMPLE);
    }

    void paintGL() override
    {
        const qreal ratio = devicePixelRatioF();
        const int framebufferWidth = qRound(width() * ratio);
        const int framebufferHeight = qRound(height() * ratio);
        glViewport(0, 0, framebufferWidth, framebufferHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glBegin(GL_QUADS);
        glColor3f(0.040f, 0.078f, 0.108f); glVertex2f(0.0f, 0.0f);
        glColor3f(0.060f, 0.105f, 0.135f); glVertex2f(1.0f, 0.0f);
        glColor3f(0.145f, 0.185f, 0.205f); glVertex2f(1.0f, 1.0f);
        glColor3f(0.105f, 0.145f, 0.165f); glVertex2f(0.0f, 1.0f);
        glEnd();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.35f, 0.48f, 0.55f, 0.14f);
        glBegin(GL_LINES);
        for (int index = 0; index <= 12; ++index) {
            const float value = index / 12.0f;
            glVertex2f(value, 0.0f); glVertex2f(value, 0.38f);
            glVertex2f(0.0f, value * 0.38f); glVertex2f(1.0f, value * 0.38f);
        }
        glEnd();
        glDisable(GL_BLEND);

        if (!m_model || m_model->isEmpty())
            return;

        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        const float nearPlane = 0.1f;
        const float farPlane = 100.0f;
        const float aspect = float(qMax(1, framebufferWidth)) / float(qMax(1, framebufferHeight));
        const float halfHeight = qTan(qDegreesToRadians(21.5f)) * nearPlane;
        const float halfWidth = halfHeight * aspect;
        glFrustum(-halfWidth, halfWidth, -halfHeight, halfHeight, nearPlane, farPlane);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(float(m_pan.x() * 0.0045), float(-m_pan.y() * 0.0045), -3.25f / m_zoom);
        glRotatef(m_rotationX, 1.0f, 0.0f, 0.0f);
        glRotatef(m_rotationY, 0.0f, 1.0f, 0.0f);

        const QVector3D lightDirection = QVector3D(-0.3f, 0.55f, 0.78f).normalized();
        if (!m_model->indices.isEmpty()) {
            glBegin(GL_TRIANGLES);
            for (quint32 index : m_model->indices) {
                if (index >= quint32(m_model->vertices.size()))
                    continue;
                const QVector3D normal = index < quint32(m_model->normals.size())
                                             ? m_model->normals.at(int(index))
                                             : QVector3D(0.0f, 0.0f, 1.0f);
                const QColor color = index < quint32(m_model->colors.size())
                                         ? m_model->colors.at(int(index))
                                         : QColor(30, 181, 126);
                const float shade = 0.56f + 0.44f * qMax(0.0f, QVector3D::dotProduct(normal.normalized(), lightDirection));
                glColor3f(color.redF() * shade, color.greenF() * shade, color.blueF() * shade);
                glNormal3f(normal.x(), normal.y(), normal.z());
                const QVector3D &vertex = m_model->vertices.at(int(index));
                glVertex3f(vertex.x(), vertex.y(), vertex.z());
            }
            glEnd();
        }

        if (m_model->indices.isEmpty()) {
            glPointSize(float(qMax(1.2, ratio * 1.15)));
            glBegin(GL_POINTS);
            for (int index = 0; index < m_model->vertices.size(); ++index) {
                const QColor color = index < m_model->colors.size()
                                         ? m_model->colors.at(index)
                                         : QColor(30, 210, 144);
                glColor3f(qMin(1.0, color.redF() * 1.12),
                          qMin(1.0, color.greenF() * 1.12),
                          qMin(1.0, color.blueF() * 1.12));
                const QVector3D &vertex = m_model->vertices.at(index);
                glVertex3f(vertex.x(), vertex.y(), vertex.z());
            }
            glEnd();
        }
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        m_lastPosition = event->localPos();
        if (event->button() == Qt::LeftButton) {
            m_rotating = true;
            setCursor(Qt::ClosedHandCursor);
        } else if (event->button() == Qt::RightButton || event->button() == Qt::MiddleButton) {
            m_panning = true;
            setCursor(Qt::SizeAllCursor);
        }
        setFocus(Qt::MouseFocusReason);
        event->accept();
    }

    void mouseMoveEvent(QMouseEvent *event) override
    {
        const QPointF delta = event->localPos() - m_lastPosition;
        if (m_rotating)
            rotateBy(float(delta.y() * 0.55), float(delta.x() * 0.55));
        else if (m_panning) {
            m_pan += delta;
            update();
        }
        m_lastPosition = event->localPos();
        event->accept();
    }

    void mouseReleaseEvent(QMouseEvent *event) override
    {
        m_rotating = false;
        m_panning = false;
        setCursor(Qt::OpenHandCursor);
        event->accept();
    }

    void mouseDoubleClickEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton)
            resetView();
        event->accept();
    }

    void wheelEvent(QWheelEvent *event) override
    {
        zoomBy(event->angleDelta().y() > 0 ? 1.11f : 0.90f);
        event->accept();
    }

    void keyPressEvent(QKeyEvent *event) override
    {
        if (event->key() == Qt::Key_Left) rotateBy(0.0f, -5.0f);
        else if (event->key() == Qt::Key_Right) rotateBy(0.0f, 5.0f);
        else if (event->key() == Qt::Key_Up) rotateBy(-5.0f, 0.0f);
        else if (event->key() == Qt::Key_Down) rotateBy(5.0f, 0.0f);
        else if (event->key() == Qt::Key_Plus || event->key() == Qt::Key_Equal) zoomBy(1.1f);
        else if (event->key() == Qt::Key_Minus) zoomBy(0.9f);
        else {
            QOpenGLWidget::keyPressEvent(event);
            return;
        }
        event->accept();
    }

private:
    ModelData *m_model = nullptr;
    float m_rotationX = -14.0f;
    float m_rotationY = 26.0f;
    float m_zoom = 0.82f;
    QPointF m_pan;
    QPointF m_lastPosition;
    bool m_rotating = false;
    bool m_panning = false;
};

EditorCanvas::EditorCanvas(QWidget *parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("EditorRoot"));
    setWindowTitle(QStringLiteral("SAM 3D 对象编辑器"));
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    setAttribute(Qt::WA_StyledBackground, true);
    setFocusPolicy(Qt::StrongFocus);
    setMinimumSize(1180, 720);

    m_sourceImage.load(QStringLiteral(":/design/sample-microbe.png"));
    m_client = new Sam3dClient(this);
    QSettings settings;
    const QString configuredEndpoint = qEnvironmentVariable(
        "SAM3D_SERVICE_URL",
        settings.value(QStringLiteral("service/endpoint"),
                       Sam3dClient::defaultEndpoint().toString()).toString());
    QString endpointError;
    if (!setServiceEndpoint(QUrl(configuredEndpoint), &endpointError))
        setServiceEndpoint(Sam3dClient::defaultEndpoint());

    buildInterface();

    connect(m_client, &Sam3dClient::readinessFinished, this,
            [this](bool ready, const QString &detail) {
        m_serviceReady = ready;
        m_serviceDetail = detail;
        applyState(false);
    });
    connect(m_client, &Sam3dClient::segmentationBusyChanged, this, [this](bool busy) {
        m_segmentBusy = busy;
        m_imageView->setInteractive(!busy && m_aiSwitch->isChecked());
        applyState(false);
    });
    connect(m_client, &Sam3dClient::segmentationFinished, this,
            [this](const QImage &mask, const QString &score, quint64 revision) {
        if (revision != m_selectionRevision)
            return;
        if (mask.size() != m_sourceImage.size()) {
            m_maskReady = false;
            m_selectionError = QStringLiteral("函数返回的 Mask 尺寸与原图不一致");
            m_imageView->clearMask();
            applyState(false);
            return;
        }
        m_maskImage = mask;
        m_maskReady = true;
        m_selectionError.clear();
        m_imageView->setMask(mask);
        if (!score.isEmpty())
            m_serviceDetail = QStringLiteral("Mask 置信度 %1").arg(score);
        applyState(false);
    });
    connect(m_client, &Sam3dClient::generationFinished,
            this, [this](const QByteArray &glb) { completeGeneration(glb); });
    connect(m_client, &Sam3dClient::requestFailed, this,
            [this](Sam3dClient::Operation operation,
                   const QString &message,
                   int,
                   quint64 revision) {
        if (operation == Sam3dClient::Operation::Segmentation) {
            if (revision != m_selectionRevision)
                return;
            m_segmentBusy = false;
            m_maskReady = false;
            m_selectionError = message;
            m_imageView->clearMask();
            applyState(false);
            showToast(QStringLiteral("选区计算失败"), message, false);
        } else if (operation == Sam3dClient::Operation::Generation) {
            m_failedBody->setText(message);
            setState(UiState::Failed);
        }
    });

    m_toastTimer.setSingleShot(true);
    connect(&m_toastTimer, &QTimer::timeout, this, [this] {
        m_savedToastVisible = false;
        m_toast->hide();
    });

    applyState(false);
    QTimer::singleShot(0, m_client, [this] { m_client->checkReady(); });
}

EditorCanvas::~EditorCanvas() = default;

bool EditorCanvas::setServiceEndpoint(const QUrl &endpoint, QString *error)
{
    if (!m_client || !m_client->setEndpoint(endpoint, error))
        return false;
    m_serviceReady = false;
    m_serviceDetail = QStringLiteral("正在检查函数服务");
    return true;
}

QUrl EditorCanvas::serviceEndpoint() const
{
    return m_client ? m_client->endpoint() : QUrl();
}

void EditorCanvas::buildInterface()
{
    setStyleSheet(QStringLiteral(R"STYLE(
        QWidget#EditorRoot {
            background: #070d19;
            color: #f0f4fb;
            font-family: "Microsoft YaHei UI", "Segoe UI";
            font-size: 13px;
        }
        QFrame#TopBar, QFrame#StatusBar, QFrame#ToolPanel, QFrame#AiPanel {
            background: #0e1524;
            border: 1px solid #39465e;
        }
        QFrame#TopBar { border-radius: 20px; }
        QFrame#StatusBar { border-radius: 14px; }
        QFrame#ToolPanel, QFrame#AiPanel { border-radius: 15px; }
        QLabel { color: #f0f4fb; background: transparent; border: none; }
        QLabel#SecondaryLabel { color: #aab5c8; }
        QLabel#MutedLabel { color: #77849c; }
        QLabel#StatusIcon { color: #3f7cff; font-size: 20px; }
        QLabel#TitleLabel { font-size: 15px; font-weight: 600; }

        QPushButton {
            min-height: 38px;
            padding: 0 16px;
            color: #dbe3f1;
            background: #202a3c;
            border: 1px solid #3b4861;
            border-radius: 10px;
            font-weight: 500;
        }
        QPushButton:hover { background: #2a3650; border-color: #52617d; color: #ffffff; }
        QPushButton:pressed { background: #182236; padding-top: 2px; padding-left: 17px; }
        QPushButton:focus { border: 1px solid #7da5ff; }
        QPushButton:disabled { background: #111827; border-color: #293348; color: #65718a; }
        QPushButton#GhostButton { background: transparent; border: none; color: #d8e0ef; padding: 0 8px; }
        QPushButton#GhostButton:hover { background: #1c2639; }
        QPushButton#ExitButton { background: #252f43; border-color: #252f43; color: #d0d8e6; }
        QPushButton#ExitButton:hover { background: #303b52; }
        QPushButton#PrimaryButton { background: #235cf0; border-color: #2f6cff; color: #ffffff; font-weight: 600; }
        QPushButton#PrimaryButton:hover { background: #3472ff; border-color: #6091ff; }
        QPushButton#PrimaryButton:pressed { background: #1848c8; }
        QPushButton#PrimaryButton:disabled { background: #10182b; border-color: #27324a; color: #687692; }
        QPushButton#AddButton:checked { background: #235cf0; border-color: #3977ff; color: #ffffff; }
        QPushButton#SubtractButton:checked { background: #eb3a43; border-color: #ff5159; color: #ffffff; }
        QPushButton#FloatingButton { background: rgba(9, 17, 31, 235); border-color: #344159; color: #dce4f2; }
        QPushButton#FloatingButton:hover { background: #1b2a40; border-color: #536582; }

        QToolButton {
            width: 42px;
            height: 42px;
            padding: 0;
            background: #202a3c;
            border: 1px solid #3a4860;
            border-radius: 10px;
        }
        QToolButton:hover { background: #2b3851; border-color: #566784; }
        QToolButton:pressed { background: #172238; padding-top: 2px; }
        QToolButton:checked { background: #173764; border-color: #2f72ff; }
        QToolButton:focus { border-color: #7da5ff; }

        QWidget#ModalShade { background: rgba(4, 9, 18, 150); }
        QFrame#ModalPage {
            background: #0f1627;
            border: 1px solid #46536b;
            border-radius: 20px;
        }
        QFrame#CreditPanel {
            background: #242d40;
            border: 1px solid #46536b;
            border-radius: 12px;
        }
        QLabel#ModalIconBlue, QLabel#ModalIconRed {
            border-radius: 24px;
            font-size: 26px;
            font-weight: 600;
        }
        QLabel#ModalIconBlue { background: #2b65e8; color: #ffffff; }
        QLabel#ModalIconRed { background: #252e42; border: 1px solid #46536b; color: #ff4a52; }
        QLabel#ModalTitle { font-size: 21px; font-weight: 650; }
        QLabel#ModalBody { color: #a3aec2; font-size: 13px; }
        QProgressBar {
            height: 8px;
            background: #2a3346;
            border: none;
            border-radius: 4px;
        }
        QProgressBar::chunk { background: #3777ff; border-radius: 4px; }

        QFrame#Toast {
            background: #11192a;
            border: 1px solid #45536d;
            border-radius: 14px;
        }
        QLabel#ToastIcon {
            background: #233047;
            border: 1px solid #465775;
            border-radius: 18px;
            color: #49d29b;
            font-size: 20px;
            font-weight: 700;
        }
        QLabel#ToastTitle { font-size: 14px; font-weight: 600; }
        QLabel#ToastDetail { color: #8794ab; font-size: 11px; }
    )STYLE"));

    m_contentLayer = new QWidget(this);
    m_contentLayer->setObjectName(QStringLiteral("ContentLayer"));
    m_contentLayer->setAttribute(Qt::WA_StyledBackground, true);

    m_imageView = new ImageSelectionView(m_contentLayer);
    m_imageView->setObjectName(QStringLiteral("ImageSelectionView"));
    m_imageView->setSourceImage(m_sourceImage);
    m_imageView->selectionChanged = [this] {
        updateSelectionState();
        requestSegmentation();
    };

    m_modelView = new ModelViewport(&m_model, m_contentLayer);
    m_modelView->hide();

    buildTopBar();
    buildStatusBar();
    buildToolBars();
    buildModal();
    buildToast();

    auto *openShortcut = new QShortcut(QKeySequence::Open, this);
    connect(openShortcut, &QShortcut::activated, this, [this] { openImage(); });
    auto *modelShortcut = new QShortcut(QKeySequence(Qt::Key_M), this);
    connect(modelShortcut, &QShortcut::activated, this, [this] { openModel(); });
    auto *fullscreenShortcut = new QShortcut(QKeySequence(Qt::Key_F11), this);
    connect(fullscreenShortcut, &QShortcut::activated, this, [this] { toggleFullscreen(); });
}

QPushButton *EditorCanvas::createTextButton(const QString &text,
                                            const QString &objectName,
                                            QWidget *parent,
                                            const QIcon &icon)
{
    auto *button = new QPushButton(icon, text, parent);
    button->setObjectName(objectName);
    button->setCursor(Qt::PointingHandCursor);
    button->setFocusPolicy(Qt::StrongFocus);
    button->setIconSize(QSize(22, 22));
    button->setAccessibleName(text);
    return button;
}

QToolButton *EditorCanvas::createToolButton(const QString &tooltip,
                                            const QIcon &icon,
                                            QWidget *parent,
                                            bool checkable)
{
    auto *button = new QToolButton(parent);
    button->setIcon(icon);
    button->setIconSize(QSize(24, 24));
    button->setToolTip(tooltip);
    button->setAccessibleName(tooltip);
    button->setCursor(Qt::PointingHandCursor);
    button->setFocusPolicy(Qt::StrongFocus);
    button->setCheckable(checkable);
    return button;
}

void EditorCanvas::buildTopBar()
{
    m_topBar = new TitleBar(this);
    m_topBar->setObjectName(QStringLiteral("TopBar"));
    m_topBar->setAttribute(Qt::WA_StyledBackground, true);

    m_backButton = createTextButton(QStringLiteral("返回"), QStringLiteral("GhostButton"),
                                    m_topBar, makeIcon(IconKind::Back, Theme::secondary));
    m_backButton->setToolTip(QStringLiteral("返回上一层"));
    connect(m_backButton, &QPushButton::clicked, this, [this] {
        if (isSplitState(m_state))
            setState(m_imageView->selectionCount() > 0 ? UiState::Selected : UiState::Waiting);
        else
            close();
    });

    m_titleLabel = new QLabel(m_topBar);
    m_titleLabel->setObjectName(QStringLiteral("TitleLabel"));
    m_titleLabel->setTextInteractionFlags(Qt::NoTextInteraction);

    m_exitButton = createTextButton(QStringLiteral("退出编辑模式"), QStringLiteral("ExitButton"), m_topBar);
    connect(m_exitButton, &QPushButton::clicked, this, &QWidget::close);

    m_saveButton = createTextButton(QStringLiteral("保存"), QStringLiteral("PrimaryButton"),
                                    m_topBar, makeIcon(IconKind::Save));
    m_saveButton->setShortcut(QKeySequence::Save);
    m_saveButton->setToolTip(QStringLiteral("保存 3D 模型 (Ctrl+S)"));
    connect(m_saveButton, &QPushButton::clicked, this, [this] {
        if (m_state == UiState::Result)
            saveModel();
        else
            showToast(QStringLiteral("请先生成 3D 模型"), QStringLiteral("完成选区后即可转换并保存"), false);
    });
}

void EditorCanvas::buildStatusBar()
{
    m_statusBar = new QFrame(this);
    m_statusBar->setObjectName(QStringLiteral("StatusBar"));
    m_statusBar->setAttribute(Qt::WA_StyledBackground, true);

    m_statusIcon = new QLabel(QStringLiteral("✦"), m_statusBar);
    m_statusIcon->setObjectName(QStringLiteral("StatusIcon"));
    m_statusIcon->setAlignment(Qt::AlignCenter);

    m_statusLabel = new QLabel(m_statusBar);
    m_statusLabel->setObjectName(QStringLiteral("SecondaryLabel"));

    m_addButton = createTextButton(QStringLiteral("＋ 增加选区"), QStringLiteral("AddButton"), m_statusBar);
    m_addButton->setCheckable(true);
    m_addButton->setChecked(true);
    connect(m_addButton, &QPushButton::clicked, this, [this] {
        m_addMode = true;
        m_addButton->setChecked(true);
        m_subtractButton->setChecked(false);
        m_imageView->setAddMode(true);
    });

    m_subtractButton = createTextButton(QStringLiteral("－ 减少选区"), QStringLiteral("SubtractButton"), m_statusBar);
    m_subtractButton->setCheckable(true);
    connect(m_subtractButton, &QPushButton::clicked, this, [this] {
        m_addMode = false;
        m_addButton->setChecked(false);
        m_subtractButton->setChecked(true);
        m_imageView->setAddMode(false);
    });

    m_generateButton = createTextButton(QStringLiteral("生成 3D 模型"), QStringLiteral("PrimaryButton"), m_statusBar);
    connect(m_generateButton, &QPushButton::clicked, this, [this] {
        if (m_maskReady)
            setState(UiState::CreditConfirm);
        else
            showToast(QStringLiteral("选区尚未完成"),
                      m_segmentBusy ? QStringLiteral("正在等待函数返回 Mask")
                                    : QStringLiteral("请先完成一次有效点选"),
                      false);
    });
}

void EditorCanvas::buildToolBars()
{
    m_leftTools = new QFrame(this);
    m_leftTools->setObjectName(QStringLiteral("ToolPanel"));
    m_leftTools->setAttribute(Qt::WA_StyledBackground, true);
    auto *leftLayout = new QHBoxLayout(m_leftTools);
    leftLayout->setContentsMargins(7, 7, 7, 7);
    leftLayout->setSpacing(5);

    const QVector<QPair<QString, IconKind>> leftDefinitions = {
        {QStringLiteral("导入显微图像 (Ctrl+O)"), IconKind::Image},
        {QStringLiteral("顺时针旋转图像"), IconKind::Rotate},
        {QStringLiteral("函数服务设置"), IconKind::Settings}
    };
    for (const auto &definition : leftDefinitions) {
        QToolButton *button = createToolButton(definition.first, makeIcon(definition.second), m_leftTools);
        m_leftToolButtons.append(button);
        leftLayout->addWidget(button);
    }
    connect(m_leftToolButtons.at(0), &QToolButton::clicked, this, [this] { openImage(); });
    connect(m_leftToolButtons.at(1), &QToolButton::clicked, this, [this] { rotateImage(); });
    connect(m_leftToolButtons.at(2), &QToolButton::clicked, this, [this] { configureService(); });

    m_centerTools = new QFrame(this);
    m_centerTools->setObjectName(QStringLiteral("ToolPanel"));
    m_centerTools->setAttribute(Qt::WA_StyledBackground, true);
    auto *centerLayout = new QHBoxLayout(m_centerTools);
    centerLayout->setContentsMargins(7, 7, 7, 7);
    centerLayout->setSpacing(5);

    const QVector<QPair<QString, IconKind>> centerDefinitions = {
        {QStringLiteral("智能捕捉"), IconKind::Target},
        {QStringLiteral("选择工具"), IconKind::Cursor},
        {QStringLiteral("线段工具"), IconKind::Line},
        {QStringLiteral("圆形笔刷"), IconKind::Circle},
        {QStringLiteral("自由套索"), IconKind::Lasso},
        {QStringLiteral("折线工具"), IconKind::Angle},
        {QStringLiteral("矩形选区"), IconKind::Rectangle},
        {QStringLiteral("椭圆选区"), IconKind::Ellipse},
        {QStringLiteral("文字标记"), IconKind::Text},
        {QStringLiteral("清空选区"), IconKind::Trash},
        {QStringLiteral("撤销上一步"), IconKind::Undo}
    };

    auto *toolGroup = new QButtonGroup(m_centerTools);
    toolGroup->setExclusive(true);
    for (int index = 0; index < centerDefinitions.size(); ++index) {
        const bool isEditingTool = index < 9;
        QToolButton *button = createToolButton(centerDefinitions.at(index).first,
                                               makeIcon(centerDefinitions.at(index).second),
                                               m_centerTools, isEditingTool);
        m_centerToolButtons.append(button);
        centerLayout->addWidget(button);
        if (isEditingTool)
            toolGroup->addButton(button, index);
        if (index > 0 && index < 9) {
            button->setEnabled(false);
            button->setToolTip(QStringLiteral("当前真实流程使用函数点选，其他绘制工具暂未接入"));
        }
    }
    m_centerToolButtons.first()->setChecked(true);
    connect(m_centerToolButtons.at(9), &QToolButton::clicked, this, [this] {
        m_imageView->clearSelections();
    });
    connect(m_centerToolButtons.at(10), &QToolButton::clicked, this, [this] {
        m_imageView->removeLastSelection();
    });

    m_aiTools = new QFrame(this);
    m_aiTools->setObjectName(QStringLiteral("AiPanel"));
    m_aiTools->setAttribute(Qt::WA_StyledBackground, true);
    m_aiLabel = new QLabel(QStringLiteral("✦  AI 捕捉微生物"), m_aiTools);
    m_aiLabel->setObjectName(QStringLiteral("SecondaryLabel"));
    m_aiSwitch = new ToggleSwitch(m_aiTools);
    connect(m_aiSwitch, &QAbstractButton::toggled, this, [this](bool enabled) {
        m_statusIcon->setText(enabled ? QStringLiteral("✦") : QStringLiteral("◌"));
        m_imageView->setInteractive(enabled && !m_segmentBusy);
        if (!enabled)
            m_client->cancelSegmentation();
        updateSelectionState();
    });

    m_downloadButton = createTextButton(QStringLiteral("下载3D"), QStringLiteral("FloatingButton"),
                                        this, makeIcon(IconKind::Download, Theme::secondary));
    m_downloadButton->setToolTip(QStringLiteral("保存 GLB 模型"));
    connect(m_downloadButton, &QPushButton::clicked, this, [this] { saveModel(); });

    m_fullscreenButton = createTextButton(QStringLiteral("全屏"), QStringLiteral("FloatingButton"),
                                          this, makeIcon(IconKind::Fullscreen, Theme::secondary));
    m_fullscreenButton->setToolTip(QStringLiteral("切换全屏 (F11)"));
    connect(m_fullscreenButton, &QPushButton::clicked, this, [this] { toggleFullscreen(); });
}

void EditorCanvas::buildModal()
{
    m_modalShade = new QWidget(this);
    m_modalShade->setObjectName(QStringLiteral("ModalShade"));
    m_modalShade->setAttribute(Qt::WA_StyledBackground, true);

    m_modalStack = new QStackedWidget(m_modalShade);
    m_modalStack->setObjectName(QStringLiteral("StateModal"));

    auto makePage = [this]() {
        auto *page = new QFrame(m_modalStack);
        page->setObjectName(QStringLiteral("ModalPage"));
        page->setAttribute(Qt::WA_StyledBackground, true);
        return page;
    };
    auto makeLabel = [](const QString &text, const QString &name, QWidget *parent) {
        auto *label = new QLabel(text, parent);
        label->setObjectName(name);
        return label;
    };

    QFrame *confirmPage = makePage();
    QLabel *confirmIcon = makeLabel(QStringLiteral("◎"), QStringLiteral("ModalIconBlue"), confirmPage);
    confirmIcon->setAlignment(Qt::AlignCenter);
    confirmIcon->setGeometry(32, 27, 48, 48);
    QLabel *confirmTitle = makeLabel(QStringLiteral("确认生成 GLB 模型"), QStringLiteral("ModalTitle"), confirmPage);
    confirmTitle->setGeometry(96, 24, 330, 32);
    QLabel *confirmSubtitle = makeLabel(QStringLiteral("将调用函数计算执行 SAM3D 重建"), QStringLiteral("ModalBody"), confirmPage);
    confirmSubtitle->setGeometry(96, 57, 300, 24);

    QFrame *creditPanel = new QFrame(confirmPage);
    creditPanel->setObjectName(QStringLiteral("CreditPanel"));
    creditPanel->setAttribute(Qt::WA_StyledBackground, true);
    creditPanel->setGeometry(32, 103, 436, 48);
    QLabel *creditCurrent = makeLabel(QStringLiteral("输出 GLB 2.0"), QStringLiteral("ModalBody"), creditPanel);
    creditCurrent->setGeometry(20, 0, 105, 48);
    QLabel *creditCost = makeLabel(QStringLiteral("种子 42"), QStringLiteral("ModalBody"), creditPanel);
    creditCost->setGeometry(158, 0, 105, 48);
    QLabel *creditAfter = makeLabel(QStringLiteral("Mask 已就绪"), QStringLiteral("ModalBody"), creditPanel);
    creditAfter->setStyleSheet(QStringLiteral("color: #4f83ff;"));
    creditAfter->setGeometry(296, 0, 120, 48);

    QLabel *confirmNote = makeLabel(QStringLiteral("确认后上传原图与当前 Mask，生成期间请保持程序运行。"),
                                    QStringLiteral("ModalBody"), confirmPage);
    confirmNote->setGeometry(32, 163, 430, 28);
    QPushButton *confirmCancel = createTextButton(QStringLiteral("取消"), QStringLiteral("ModalButton"), confirmPage);
    confirmCancel->setGeometry(172, 215, 142, 48);
    QPushButton *confirmAccept = createTextButton(QStringLiteral("确认转换"), QStringLiteral("PrimaryButton"), confirmPage);
    confirmAccept->setGeometry(326, 215, 142, 48);
    connect(confirmCancel, &QPushButton::clicked, this, [this] {
        setState(m_imageView->selectionCount() > 0 ? UiState::Selected : UiState::Waiting);
    });
    connect(confirmAccept, &QPushButton::clicked, this, [this] {
        m_demoStateLocked = false;
        beginGeneration();
    });

    QFrame *generatingPage = makePage();
    QLabel *generatingIcon = makeLabel(QStringLiteral("◔"), QStringLiteral("ModalIconBlue"), generatingPage);
    generatingIcon->setAlignment(Qt::AlignCenter);
    generatingIcon->setGeometry(226, 28, 48, 48);
    QLabel *generatingTitle = makeLabel(QStringLiteral("正在生成 3D 模型"), QStringLiteral("ModalTitle"), generatingPage);
    generatingTitle->setAlignment(Qt::AlignCenter);
    generatingTitle->setGeometry(80, 94, 340, 34);
    QLabel *generatingBody = makeLabel(QStringLiteral("正在重建微生物的细节与深度信息"), QStringLiteral("ModalBody"), generatingPage);
    generatingBody->setAlignment(Qt::AlignCenter);
    generatingBody->setGeometry(70, 135, 360, 26);
    m_generationProgress = new QProgressBar(generatingPage);
    m_generationProgress->setTextVisible(false);
    m_generationProgress->setRange(0, 0);
    m_generationProgress->setGeometry(88, 181, 324, 8);
    QLabel *generatingNote = makeLabel(QStringLiteral("函数计算正在推理，请勿关闭程序"), QStringLiteral("ModalBody"), generatingPage);
    generatingNote->setAlignment(Qt::AlignCenter);
    generatingNote->setGeometry(100, 205, 300, 26);

    QFrame *failedPage = makePage();
    QLabel *failedIcon = makeLabel(QStringLiteral("!"), QStringLiteral("ModalIconRed"), failedPage);
    failedIcon->setAlignment(Qt::AlignCenter);
    failedIcon->setGeometry(226, 28, 48, 48);
    QLabel *failedTitle = makeLabel(QStringLiteral("3D 模型生成失败"), QStringLiteral("ModalTitle"), failedPage);
    failedTitle->setAlignment(Qt::AlignCenter);
    failedTitle->setGeometry(75, 94, 350, 34);
    m_failedBody = makeLabel(QStringLiteral("生成过程未能完成，请检查网络后重新尝试"), QStringLiteral("ModalBody"), failedPage);
    m_failedBody->setAlignment(Qt::AlignCenter);
    m_failedBody->setGeometry(60, 136, 380, 28);
    QPushButton *failedBack = createTextButton(QStringLiteral("返回编辑"), QStringLiteral("ModalButton"), failedPage);
    failedBack->setGeometry(124, 186, 120, 44);
    QPushButton *failedRetry = createTextButton(QStringLiteral("重新生成"), QStringLiteral("PrimaryButton"), failedPage);
    failedRetry->setGeometry(256, 186, 120, 44);
    connect(failedBack, &QPushButton::clicked, this, [this] {
        setState(m_imageView->selectionCount() > 0 ? UiState::Selected : UiState::Waiting);
    });
    connect(failedRetry, &QPushButton::clicked, this, [this] {
        m_demoStateLocked = false;
        beginGeneration();
    });

    m_modalStack->addWidget(confirmPage);
    m_modalStack->addWidget(generatingPage);
    m_modalStack->addWidget(failedPage);

    m_modalOpacity = new QGraphicsOpacityEffect(m_modalStack);
    m_modalOpacity->setOpacity(1.0);
    m_modalStack->setGraphicsEffect(m_modalOpacity);
    m_modalAnimation = new QPropertyAnimation(m_modalOpacity, "opacity", this);
    m_modalAnimation->setDuration(170);
    m_modalAnimation->setEasingCurve(QEasingCurve::OutCubic);
    m_modalShade->hide();
}

void EditorCanvas::buildToast()
{
    m_toast = new QFrame(this);
    m_toast->setObjectName(QStringLiteral("Toast"));
    m_toast->setAttribute(Qt::WA_StyledBackground, true);

    QLabel *toastIcon = new QLabel(QStringLiteral("✓"), m_toast);
    toastIcon->setObjectName(QStringLiteral("ToastIcon"));
    toastIcon->setAlignment(Qt::AlignCenter);
    toastIcon->setGeometry(16, 14, 36, 36);

    m_toastTitle = new QLabel(QStringLiteral("已保存为模型"), m_toast);
    m_toastTitle->setObjectName(QStringLiteral("ToastTitle"));
    m_toastDetail = new QLabel(m_toast);
    m_toastDetail->setObjectName(QStringLiteral("ToastDetail"));
    m_toastAction = createTextButton(QStringLiteral("点击查看"), QStringLiteral("PrimaryButton"), m_toast);
    m_toastAction->setGeometry(260, 12, 88, 40);
    connect(m_toastAction, &QPushButton::clicked, this, [this] {
        m_savedToastVisible = false;
        m_toast->hide();
        if (m_modelView->isVisible())
            m_modelView->setFocus(Qt::OtherFocusReason);
    });
    m_toast->hide();
}

void EditorCanvas::layoutInterface()
{
    const int canvasWidth = width();
    const int canvasHeight = height();
    const bool split = isSplitState(m_state);
    const int half = canvasWidth / 2;

    m_contentLayer->setGeometry(rect());
    m_imageView->setSplit(split);
    m_imageView->setGeometry(0, 0, split ? half : canvasWidth, canvasHeight);
    m_modelView->setGeometry(half, 0, canvasWidth - half, canvasHeight);
    m_modelView->setVisible(split);

    m_topBar->setGeometry(40, 28, canvasWidth - 80, 64);
    m_backButton->setGeometry(12, 10, 102, 44);
    m_titleLabel->setGeometry(130, 8, 400, 48);
    m_exitButton->setGeometry(m_topBar->width() / 2 - 72, 10, 144, 42);
    m_saveButton->setGeometry(m_topBar->width() - 94, 10, 82, 44);

    const int editorLeft = (canvasWidth - 980) / 2;
    const int statusTop = canvasHeight - 159;
    const int toolsTop = canvasHeight - 94;
    m_statusBar->setGeometry(editorLeft, statusTop, 980, 53);
    m_statusIcon->setGeometry(12, 7, 32, 39);
    m_statusLabel->setGeometry(46, 7, 470, 39);
    m_addButton->setGeometry(527, 6, 132, 41);
    m_subtractButton->setGeometry(667, 6, 132, 41);
    m_generateButton->setGeometry(807, 6, 164, 41);

    m_leftTools->setGeometry(editorLeft, toolsTop, 166, 60);
    m_centerTools->setGeometry(editorLeft + 177, toolsTop, 580, 60);
    m_aiTools->setGeometry(editorLeft + 769, toolsTop, 210, 60);
    m_aiLabel->setGeometry(14, 8, 140, 44);
    m_aiSwitch->move(153, 17);

    m_downloadButton->setGeometry(half + 222, canvasHeight - 216, 96, 41);
    m_fullscreenButton->setGeometry(half + 326, canvasHeight - 216, 92, 41);

    m_modalShade->setGeometry(rect());
    m_modalStack->setGeometry((canvasWidth - 500) / 2, (canvasHeight - 286) / 2, 500, 286);

    m_toast->setGeometry((canvasWidth - 360) / 2, 116, 360, 64);
    const bool hasDetail = !m_toastDetail->text().isEmpty();
    m_toastTitle->setGeometry(64, hasDetail ? 10 : 0, 188, hasDetail ? 24 : 64);
    m_toastDetail->setGeometry(64, 32, 188, 20);

    m_contentLayer->lower();
    m_imageView->lower();
    if (split)
        m_modelView->raise();
    m_topBar->raise();
    m_statusBar->raise();
    m_leftTools->raise();
    m_centerTools->raise();
    m_aiTools->raise();
    m_downloadButton->raise();
    m_fullscreenButton->raise();
    if (m_toast->isVisible())
        m_toast->raise();
    if (m_modalShade->isVisible())
        m_modalShade->raise();
}

void EditorCanvas::applyState(bool animateModal)
{
    const bool split = isSplitState(m_state);
    m_titleLabel->setText(split
                              ? QStringLiteral("编辑: %1  ·  3D 预览").arg(m_imageName)
                              : QStringLiteral("编辑: %1").arg(m_imageName));

    const int count = m_imageView->positiveCount();
    if (count == 0) {
        if (!m_serviceDetail.isEmpty() && !m_serviceReady)
            m_statusLabel->setText(QStringLiteral("函数服务：%1").arg(m_serviceDetail));
        else
            m_statusLabel->setText(m_aiSwitch->isChecked()
                                       ? QStringLiteral("请点击画面中的微生物，左键增加，右键减少")
                                       : QStringLiteral("AI 捕捉已关闭"));
    } else if (!m_selectionError.isEmpty()) {
        m_statusLabel->setText(QStringLiteral("选区计算失败：%1").arg(m_selectionError));
    } else if (m_segmentBusy) {
        m_statusLabel->setText(QStringLiteral("正在通过函数计算更新选区 · POST /segment"));
    } else if (m_maskReady) {
        m_statusLabel->setText(QStringLiteral("函数 Mask 已更新，可继续增加或减少选区"));
    } else {
        m_statusLabel->setText(QStringLiteral("等待函数返回选区 Mask"));
    }

    m_generateButton->setEnabled(m_maskReady && m_state != UiState::Generating);
    m_addButton->setEnabled(m_state != UiState::Generating);
    m_subtractButton->setEnabled(m_state != UiState::Generating);
    m_addButton->setChecked(m_addMode);
    m_subtractButton->setChecked(!m_addMode);
    m_imageView->setAddMode(m_addMode);
    m_imageView->setInteractive(m_aiSwitch->isChecked()
                                && !m_segmentBusy
                                && m_state != UiState::Generating);

    const bool hasResult = m_state == UiState::Result && !m_model.isEmpty();
    m_downloadButton->setVisible(hasResult);
    m_fullscreenButton->setVisible(hasResult);

    const bool modalVisible = m_state == UiState::CreditConfirm
                              || m_state == UiState::Generating
                              || m_state == UiState::Failed;
    if (modalVisible) {
        if (m_state == UiState::CreditConfirm)
            m_modalStack->setCurrentIndex(0);
        else if (m_state == UiState::Generating)
            m_modalStack->setCurrentIndex(1);
        else
            m_modalStack->setCurrentIndex(2);

        m_modalShade->show();
        if (animateModal) {
            m_modalAnimation->stop();
            m_modalAnimation->setStartValue(0.0);
            m_modalAnimation->setEndValue(1.0);
            m_modalAnimation->start();
        } else {
            m_modalOpacity->setOpacity(1.0);
        }
    } else {
        m_modalAnimation->stop();
        m_modalShade->hide();
    }

    m_toast->setVisible(m_savedToastVisible);
    layoutInterface();
    m_modelView->update();
}

void EditorCanvas::setState(UiState state, bool animateModal)
{
    if (m_state == UiState::Generating && state != UiState::Generating && m_client)
        m_client->cancelGeneration();
    m_state = state;
    applyState(animateModal);
}

void EditorCanvas::updateSelectionState()
{
    if (m_state == UiState::Waiting || m_state == UiState::Selected)
        m_state = m_imageView->selectionCount() > 0 ? UiState::Selected : UiState::Waiting;
    applyState(false);
}

void EditorCanvas::openImage()
{
    const QString initial = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    const QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("导入显微图像"), initial,
                                                          QStringLiteral("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
    if (fileName.isEmpty())
        return;

    QImage image(fileName);
    if (image.isNull()) {
        showToast(QStringLiteral("无法读取所选图像"), QStringLiteral("请选择有效的 PNG、JPG、BMP 或 TIFF 文件"), false);
        return;
    }

    m_client->cancelSegmentation();
    m_client->cancelGeneration();
    ++m_selectionRevision;
    m_maskReady = false;
    m_maskImage = QImage();
    m_selectionError.clear();
    m_model.clear();
    m_sourceImage = image;
    m_imageView->setSourceImage(m_sourceImage);
    m_imageView->clearSelections(false);
    m_imageName = QFileInfo(fileName).completeBaseName();
    setState(UiState::Waiting, false);
    showToast(QStringLiteral("图像已导入"), m_imageName, false);
}

void EditorCanvas::rotateImage()
{
    if (m_sourceImage.isNull())
        return;
    QTransform transform;
    transform.rotate(90.0);
    m_client->cancelSegmentation();
    ++m_selectionRevision;
    m_maskReady = false;
    m_maskImage = QImage();
    m_selectionError.clear();
    m_sourceImage = m_sourceImage.transformed(transform, Qt::SmoothTransformation);
    m_imageView->setSourceImage(m_sourceImage);
    m_imageView->clearSelections(false);
    setState(UiState::Waiting, false);
    showToast(QStringLiteral("图像已旋转"), QStringLiteral("顺时针旋转 90°"), false);
}

void EditorCanvas::openModel()
{
    const QString initial = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    const QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("导入 3D 模型"), initial,
                                                          QStringLiteral("GLB 2.0 模型 (*.glb)"));
    if (fileName.isEmpty())
        return;

    QString error;
    if (!m_model.load(fileName, &error)) {
        setState(UiState::Failed);
        showToast(QStringLiteral("模型导入失败"), error, false);
        return;
    }

    m_modelName = QFileInfo(fileName).completeBaseName();
    m_modelView->resetView();
    setState(UiState::Result, false);
    showToast(QStringLiteral("3D 模型已导入"), m_modelName, false);
}

void EditorCanvas::configureService()
{
    bool accepted = false;
    const QString value = QInputDialog::getText(
        this,
        QStringLiteral("函数服务设置"),
        QStringLiteral("SAM3 / SAM3D 组合函数地址"),
        QLineEdit::Normal,
        serviceEndpoint().toString(),
        &accepted);
    if (!accepted)
        return;

    QString error;
    if (!setServiceEndpoint(QUrl(value.trimmed()), &error)) {
        showToast(QStringLiteral("函数地址无效"), error, false);
        return;
    }
    QSettings().setValue(QStringLiteral("service/endpoint"), serviceEndpoint().toString());
    applyState(false);
    m_client->checkReady();
    showToast(QStringLiteral("函数地址已保存"), serviceEndpoint().host(), false);
}

void EditorCanvas::saveModel()
{
    if (m_model.isEmpty()) {
        showToast(QStringLiteral("没有可保存的 3D 模型"), QString(), false);
        return;
    }

    const QString initial = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                            + QStringLiteral("/sam-3d-model.glb");
    const QString fileName = QFileDialog::getSaveFileName(this, QStringLiteral("保存 3D 模型"), initial,
                                                          QStringLiteral("GLB 2.0 模型 (*.glb)"));
    if (fileName.isEmpty())
        return;

    QString output = fileName;
    if (!output.endsWith(QStringLiteral(".glb"), Qt::CaseInsensitive))
        output += QStringLiteral(".glb");

    QString error;
    if (!m_model.saveGlb(output, &error)) {
        showToast(QStringLiteral("模型保存失败"), error, false);
        return;
    }
    showToast(QStringLiteral("已保存为模型"), QFileInfo(output).fileName(), true);
}

void EditorCanvas::requestSegmentation()
{
    ++m_selectionRevision;
    m_maskReady = false;
    m_maskImage = QImage();
    m_selectionError.clear();
    m_imageView->clearMask();

    const QVector<Sam3dClient::Point> points = m_imageView->points();
    bool hasPositive = false;
    for (const Sam3dClient::Point &point : points) {
        if (point.label == 1) {
            hasPositive = true;
            break;
        }
    }
    if (m_demoStateLocked || points.isEmpty() || !hasPositive) {
        m_client->cancelSegmentation();
        applyState(false);
        return;
    }

    m_client->segment(m_sourceImage, points, m_selectionRevision);
}

void EditorCanvas::beginGeneration()
{
    if (!m_maskReady || m_maskImage.isNull()) {
        showToast(QStringLiteral("无法开始生成"), QStringLiteral("当前选区还没有有效 Mask"), false);
        return;
    }
    m_failedBody->setText(QStringLiteral("生成过程未能完成，请检查网络后重新尝试"));
    setState(UiState::Generating);
    m_client->generate(m_sourceImage, m_maskImage, 42);
}

void EditorCanvas::completeGeneration(const QByteArray &glb)
{
    QString error;
    if (!m_model.loadGlbData(glb, &error)) {
        m_failedBody->setText(QStringLiteral("GLB 解析失败：%1").arg(error));
        setState(UiState::Failed);
        return;
    }
    m_modelName = QStringLiteral("%1 · SAM3D GLB").arg(m_imageName);
    m_modelView->resetView();
    setState(UiState::Result);
}

void EditorCanvas::toggleFullscreen()
{
    isFullScreen() ? showNormal() : showFullScreen();
}

void EditorCanvas::showToast(const QString &title, const QString &detail, bool showAction)
{
    m_savedToastVisible = true;
    m_toastTitle->setText(title);
    m_toastDetail->setText(detail);
    m_toastAction->setVisible(showAction);
    m_toast->show();
    layoutInterface();
    m_toast->raise();
    m_toastTimer.start(3200);
}

void EditorCanvas::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    layoutInterface();
}

void EditorCanvas::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Escape) {
        if (isFullScreen())
            showNormal();
        else if (isSplitState(m_state))
            setState(m_imageView->selectionCount() > 0 ? UiState::Selected : UiState::Waiting);
        else
            close();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Left) m_modelView->rotateBy(0.0f, -5.0f);
    else if (event->key() == Qt::Key_Right) m_modelView->rotateBy(0.0f, 5.0f);
    else if (event->key() == Qt::Key_Up) m_modelView->rotateBy(-5.0f, 0.0f);
    else if (event->key() == Qt::Key_Down) m_modelView->rotateBy(5.0f, 0.0f);
    else if (event->key() == Qt::Key_Plus || event->key() == Qt::Key_Equal) m_modelView->zoomBy(1.1f);
    else if (event->key() == Qt::Key_Minus) m_modelView->zoomBy(0.9f);
    else {
        QWidget::keyPressEvent(event);
        return;
    }
    event->accept();
}

void EditorCanvas::setDemoState(const QString &stateName)
{
    const QString name = stateName.trimmed().toLower();
    m_client->cancelSegmentation();
    m_client->cancelGeneration();
    m_toastTimer.stop();
    m_demoStateLocked = true;
    m_savedToastVisible = false;
    m_addMode = true;
    m_imageView->setDemoSelections(name != QStringLiteral("waiting"));
    m_maskReady = name != QStringLiteral("waiting");
    m_selectionError.clear();
    if (name != QStringLiteral("result") && name != QStringLiteral("saved"))
        m_model.clear();

    if (name == QStringLiteral("confirm"))
        m_state = UiState::CreditConfirm;
    else if (name == QStringLiteral("generating"))
        m_state = UiState::Generating;
    else if (name == QStringLiteral("failed"))
        m_state = UiState::Failed;
    else if (name == QStringLiteral("result") || name == QStringLiteral("saved")) {
        m_state = UiState::Result;
        m_model.createOrganicSample();
        m_modelName = QStringLiteral("叶片表皮 · 微生物重建");
        if (name == QStringLiteral("saved")) {
            m_savedToastVisible = true;
            m_toastTitle->setText(QStringLiteral("已保存为模型"));
            m_toastDetail->clear();
            m_toastAction->show();
        }
    } else if (name == QStringLiteral("selected")) {
        m_state = UiState::Selected;
    } else {
        m_state = UiState::Waiting;
        m_maskReady = false;
        m_imageView->setDemoSelections(false);
    }
    applyState(false);
}
