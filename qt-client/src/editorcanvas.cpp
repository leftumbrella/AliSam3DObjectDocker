#include "editorcanvas.h"

#include <QApplication>
#include <QCursor>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFontMetrics>
#include <QKeySequence>
#include <QKeyEvent>
#include <QLineF>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QStandardPaths>
#include <QToolTip>
#include <QWheelEvent>
#include <QWindow>
#include <QtMath>

#include <algorithm>

namespace Theme {

const QColor canvas(7, 13, 25);
const QColor surface(13, 20, 35, 248);
const QColor surfaceRaised(20, 29, 47, 250);
const QColor surfaceSoft(28, 39, 59, 236);
const QColor border(57, 70, 94);
const QColor borderSoft(42, 54, 74);
const QColor text(244, 247, 252);
const QColor textSecondary(168, 179, 198);
const QColor textMuted(112, 126, 148);
const QColor primary(42, 104, 255);
const QColor primaryHover(67, 124, 255);
const QColor mask(238, 50, 66);
const QColor success(20, 193, 127);
const QColor warning(255, 179, 71);
const QColor danger(255, 82, 99);

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

constexpr qreal kDesignWidth = 1280.0;
constexpr qreal kDesignHeight = 800.0;

void roundedPanel(QPainter &painter,
                  const QRectF &rect,
                  qreal radius,
                  const QColor &fill,
                  const QColor &stroke = Qt::transparent,
                  qreal strokeWidth = 1.0)
{
    painter.setPen(stroke.alpha() > 0 ? QPen(stroke, strokeWidth) : Qt::NoPen);
    painter.setBrush(fill);
    painter.drawRoundedRect(rect, radius, radius);
}

void centeredText(QPainter &painter,
                  const QRectF &rect,
                  const QString &text,
                  const QColor &color,
                  const QFont &font,
                  Qt::Alignment alignment = Qt::AlignCenter)
{
    painter.setPen(color);
    painter.setFont(font);
    painter.drawText(rect, alignment, text);
}

} // namespace

EditorCanvas::EditorCanvas(QWidget *parent)
    : QOpenGLWidget(parent)
{
    setWindowTitle(QStringLiteral("SAM 3D 对象编辑器"));
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    setAttribute(Qt::WA_OpaquePaintEvent);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setMinimumSize(960, 600);

    m_sourceImage.load(QStringLiteral(":/design/sample-microbe.png"));
    m_templateSelected.load(QStringLiteral(":/design/sample-selected.png"));
    m_templateConfirm.load(QStringLiteral(":/design/sample-confirm.png"));
    m_templateGenerating.load(QStringLiteral(":/design/sample-generating.png"));
    m_templateFailed.load(QStringLiteral(":/design/sample-failed.png"));
    m_templateSaved.load(QStringLiteral(":/design/sample-saved.png"));
    m_model.createOrganicSample();

    m_generationTimer.setSingleShot(true);
    connect(&m_generationTimer, &QTimer::timeout, this, [this] { completeGeneration(); });

    m_spinnerTimer.setInterval(82);
    connect(&m_spinnerTimer, &QTimer::timeout, this, [this] {
        m_loadingPhase = (m_loadingPhase + 1) % 12;
        if (m_state == UiState::Generating)
            update();
    });
    m_spinnerTimer.start();

    m_toastTimer.setSingleShot(true);
    connect(&m_toastTimer, &QTimer::timeout, this, [this] {
        m_savedToastVisible = false;
        update();
    });

    m_messageTimer.setSingleShot(true);
    connect(&m_messageTimer, &QTimer::timeout, this, [this] {
        m_transientMessage.clear();
        update();
    });
}

EditorCanvas::~EditorCanvas()
{
    if (m_overlayTexture != 0 && context()) {
        makeCurrent();
        glDeleteTextures(1, &m_overlayTexture);
        m_overlayTexture = 0;
        doneCurrent();
    }
}

QPointF EditorCanvas::toDesign(const QPointF &point) const
{
    return QPointF(point.x() * kDesignWidth / qMax(1, width()),
                   point.y() * kDesignHeight / qMax(1, height()));
}

QRectF EditorCanvas::designRect(qreal x, qreal y, qreal widthValue, qreal heightValue) const
{
    return QRectF(x, y, widthValue, heightValue);
}

bool EditorCanvas::isResultState() const
{
    return m_state == UiState::Result;
}

void EditorCanvas::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(Theme::canvas.redF(), Theme::canvas.greenF(), Theme::canvas.blueF(), 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_MULTISAMPLE);
    glGenTextures(1, &m_overlayTexture);
    glBindTexture(GL_TEXTURE_2D, m_overlayTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void EditorCanvas::resizeGL(int widthValue, int heightValue)
{
    Q_UNUSED(widthValue)
    Q_UNUSED(heightValue)
}

void EditorCanvas::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (isResultState())
        renderModel();

    QImage overlay(qRound(kDesignWidth), qRound(kDesignHeight), QImage::Format_RGBA8888);
    overlay.fill(Qt::transparent);
    {
        QPainter painter(&overlay);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setRenderHint(QPainter::TextAntialiasing, true);
        painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
        drawScene(painter);
    }

    const qreal ratio = devicePixelRatioF();
    glViewport(0, 0, qRound(width() * ratio), qRound(height() * ratio));
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_overlayTexture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, overlay.width(), overlay.height(), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, overlay.constBits());

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 1.0f);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void EditorCanvas::renderModel()
{
    if (m_model.isEmpty())
        return;

    const qreal ratio = devicePixelRatioF();
    const int framebufferWidth = qRound(width() * ratio);
    const int framebufferHeight = qRound(height() * ratio);
    const int viewportX = framebufferWidth / 2;
    const int viewportWidth = framebufferWidth - viewportX;
    glViewport(viewportX, 0, viewportWidth, framebufferHeight);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glBegin(GL_QUADS);
    glColor3f(0.075f, 0.115f, 0.135f); glVertex2f(0.0f, 0.0f);
    glColor3f(0.075f, 0.115f, 0.135f); glVertex2f(1.0f, 0.0f);
    glColor3f(0.225f, 0.270f, 0.295f); glVertex2f(1.0f, 1.0f);
    glColor3f(0.225f, 0.270f, 0.295f); glVertex2f(0.0f, 1.0f);
    glEnd();
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const float nearPlane = 0.1f;
    const float farPlane = 100.0f;
    const float aspect = float(qMax(1, viewportWidth)) / float(qMax(1, framebufferHeight));
    const float halfHeight = qTan(qDegreesToRadians(21.5f)) * nearPlane;
    const float halfWidth = halfHeight * aspect;
    glFrustum(-halfWidth, halfWidth, -halfHeight, halfHeight, nearPlane, farPlane);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(float(m_pan.x() * 0.0045), float(-m_pan.y() * 0.0045), -3.25f / m_zoom);
    glRotatef(m_rotationX, 1.0f, 0.0f, 0.0f);
    glRotatef(m_rotationY, 0.0f, 1.0f, 0.0f);

    const QVector3D lightDirection = QVector3D(-0.3f, 0.55f, 0.78f).normalized();
    if (!m_model.indices.isEmpty()) {
        glBegin(GL_TRIANGLES);
        for (quint32 index : m_model.indices) {
            if (index >= quint32(m_model.vertices.size()))
                continue;
            const QVector3D normal = index < quint32(m_model.normals.size())
                                         ? m_model.normals.at(int(index))
                                         : QVector3D(0.0f, 0.0f, 1.0f);
            const QColor color = index < quint32(m_model.colors.size())
                                     ? m_model.colors.at(int(index))
                                     : QColor(12, 180, 120);
            const float shade = 0.58f + 0.42f * qMax(0.0f, QVector3D::dotProduct(normal.normalized(), lightDirection));
            glColor3f(color.redF() * shade, color.greenF() * shade, color.blueF() * shade);
            glNormal3f(normal.x(), normal.y(), normal.z());
            const QVector3D &vertex = m_model.vertices.at(int(index));
            glVertex3f(vertex.x(), vertex.y(), vertex.z());
        }
        glEnd();
    }

    glPointSize(float(qMax(1.3, ratio * 1.25)));
    glBegin(GL_POINTS);
    for (int index = 0; index < m_model.vertices.size(); ++index) {
        const QColor color = index < m_model.colors.size() ? m_model.colors.at(index) : QColor(26, 219, 147);
        glColor3f(qMin(1.0, color.redF() * 1.16),
                  qMin(1.0, color.greenF() * 1.14),
                  qMin(1.0, color.blueF() * 1.15));
        const QVector3D &vertex = m_model.vertices.at(index);
        glVertex3f(vertex.x(), vertex.y(), vertex.z());
    }
    glEnd();
}

void EditorCanvas::drawScene(QPainter &painter)
{
    if (m_usingBundledSample) {
        if (m_state == UiState::Result && !m_savedToastVisible && !m_templateSaved.isNull()) {
            painter.drawImage(QRectF(0, 0, 640, 800), m_templateSelected, QRectF(0, 0, 640, 800));
            const auto drawTemplatePiece = [&painter, this](const QRectF &rect, qreal radius) {
                painter.save();
                QPainterPath clip;
                clip.addRoundedRect(rect, radius, radius);
                painter.setClipPath(clip);
                painter.drawImage(rect, m_templateSaved, rect);
                painter.restore();
            };
            drawTemplatePiece(QRectF(40, 28, 1200, 64), 20);
            drawTemplatePiece(QRectF(150, 641, 980, 53), 14);
            drawTemplatePiece(QRectF(150, 706, 166, 60), 15);
            drawTemplatePiece(QRectF(327, 706, 580, 60), 15);
            drawTemplatePiece(QRectF(919, 706, 210, 60), 15);
            drawTemplatePiece(QRectF(862, 584, 96, 41), 10);
            drawTemplatePiece(QRectF(966, 584, 92, 41), 10);
            return;
        }
        const QImage *designState = nullptr;
        if (m_state == UiState::Waiting)
            designState = &m_sourceImage;
        else if (m_state == UiState::Selected)
            designState = &m_templateSelected;
        else if (m_state == UiState::CreditConfirm)
            designState = &m_templateConfirm;
        else if (m_state == UiState::Generating)
            designState = &m_templateGenerating;
        else if (m_state == UiState::Failed)
            designState = &m_templateFailed;
        else if (m_state == UiState::Result && m_savedToastVisible)
            designState = &m_templateSaved;

        if (designState && !designState->isNull()) {
            painter.drawImage(QRectF(0, 0, kDesignWidth, kDesignHeight), *designState);
            return;
        }
    }

    if (isResultState()) {
        painter.fillRect(designRect(0, 0, 640, 800), Theme::canvas);
        drawImageCover(painter, designRect(0, 0, 640, 800));
        painter.fillRect(designRect(0, 0, 640, 800), QColor(4, 9, 18, 34));
        drawSelections(painter, designRect(0, 0, 640, 800));
        painter.fillRect(designRect(639, 0, 1, 800), Theme::borderSoft);
    } else {
        painter.fillRect(designRect(0, 0, 1280, 800), Theme::canvas);
        drawImageCover(painter, designRect(0, 0, 1280, 800));
        painter.fillRect(designRect(0, 0, 1280, 800), QColor(4, 9, 18, 46));
        drawSelections(painter, designRect(0, 0, 1280, 800));
    }

    drawTopBar(painter);
    if (isResultState()) {
        drawResultOverlay(painter);
        drawBottomEditor(painter);
    } else {
        drawBottomEditor(painter);
    }

    if (m_state == UiState::CreditConfirm || m_state == UiState::Generating || m_state == UiState::Failed)
        drawModal(painter);
    if (m_savedToastVisible)
        drawSavedToast(painter);

    if (!m_transientMessage.isEmpty()) {
        const QRectF messageRect(460, 112, 360, 48);
        roundedPanel(painter, messageRect, 12, QColor(19, 28, 46, 248), Theme::border);
        centeredText(painter, messageRect.adjusted(18, 0, -18, 0), m_transientMessage, Theme::text,
                     Theme::font(14, QFont::DemiBold));
    }
}

void EditorCanvas::drawImageCover(QPainter &painter, const QRectF &target)
{
    if (m_sourceImage.isNull()) {
        painter.fillRect(target, QColor(36, 48, 63));
        return;
    }

    const qreal imageAspect = qreal(m_sourceImage.width()) / qreal(m_sourceImage.height());
    const qreal targetAspect = target.width() / target.height();
    QRectF source(0, 0, m_sourceImage.width(), m_sourceImage.height());
    if (imageAspect > targetAspect) {
        const qreal sourceWidth = m_sourceImage.height() * targetAspect;
        source.setLeft((m_sourceImage.width() - sourceWidth) * 0.5);
        source.setWidth(sourceWidth);
    } else {
        const qreal sourceHeight = m_sourceImage.width() / targetAspect;
        source.setTop((m_sourceImage.height() - sourceHeight) * 0.5);
        source.setHeight(sourceHeight);
    }
    painter.drawImage(target, m_sourceImage, source);
}

void EditorCanvas::drawSelections(QPainter &painter, const QRectF &clipRect)
{
    painter.save();
    painter.setClipRect(clipRect);
    for (int index = 0; index < m_marks.size(); ++index) {
        const SelectionMark &mark = m_marks.at(index);
        QPainterPath path;
        const qreal wobble = 8.0 + (index % 3) * 3.0;
        path.moveTo(mark.center.x() - mark.radiusX, mark.center.y() + wobble * 0.2);
        path.cubicTo(mark.center.x() - mark.radiusX * 0.88, mark.center.y() - mark.radiusY * 0.84,
                     mark.center.x() - mark.radiusX * 0.31, mark.center.y() - mark.radiusY - wobble,
                     mark.center.x() + wobble, mark.center.y() - mark.radiusY * 0.94);
        path.cubicTo(mark.center.x() + mark.radiusX * 0.78, mark.center.y() - mark.radiusY * 0.78,
                     mark.center.x() + mark.radiusX + wobble, mark.center.y() - mark.radiusY * 0.1,
                     mark.center.x() + mark.radiusX * 0.89, mark.center.y() + wobble);
        path.cubicTo(mark.center.x() + mark.radiusX * 0.72, mark.center.y() + mark.radiusY * 0.82,
                     mark.center.x() + mark.radiusX * 0.2, mark.center.y() + mark.radiusY + wobble,
                     mark.center.x() - wobble, mark.center.y() + mark.radiusY * 0.92);
        path.cubicTo(mark.center.x() - mark.radiusX * 0.77, mark.center.y() + mark.radiusY * 0.72,
                     mark.center.x() - mark.radiusX - wobble, mark.center.y() + mark.radiusY * 0.17,
                     mark.center.x() - mark.radiusX, mark.center.y() + wobble * 0.2);
        path.closeSubpath();

        if (mark.positive) {
            painter.setBrush(QColor(Theme::mask.red(), Theme::mask.green(), Theme::mask.blue(), 132));
            painter.setPen(QPen(QColor(255, 86, 98, 220), 1.4));
            painter.drawPath(path);
        } else {
            painter.setBrush(QColor(5, 11, 20, 154));
            painter.setPen(QPen(QColor(255, 181, 71, 230), 1.5, Qt::DashLine));
            painter.drawPath(path);
            drawIcon(painter, ToolIcon::Minus, mark.center, Theme::warning, 24);
        }
    }
    painter.restore();
}

void EditorCanvas::drawTopBar(QPainter &painter)
{
    const QRectF bar(40, 28, 1200, 64);
    roundedPanel(painter, bar, 20, Theme::surface, Theme::border, 1.0);

    painter.setPen(QPen(Theme::borderSoft, 1));
    painter.drawLine(QPointF(146, 44), QPointF(146, 76));

    drawIcon(painter, ToolIcon::ArrowLeft, QPointF(88, 60), Theme::textSecondary, 18);
    centeredText(painter, QRectF(105, 39, 38, 42), QStringLiteral("返回"), Theme::textSecondary,
                 Theme::font(12, QFont::DemiBold), Qt::AlignVCenter | Qt::AlignLeft);

    centeredText(painter, QRectF(171, 36, 352, 48),
                 isResultState() ? QStringLiteral("编辑：%1 · 3D 预览").arg(m_imageName)
                                 : QStringLiteral("编辑：%1").arg(m_imageName),
                 Theme::text, Theme::font(14, QFont::DemiBold), Qt::AlignVCenter | Qt::AlignLeft);

    const QRectF exitRect(638, 39, 144, 42);
    roundedPanel(painter, exitRect, 10, Theme::surfaceSoft, Theme::borderSoft);
    centeredText(painter, exitRect, QStringLiteral("退出编辑模式"), Theme::textSecondary,
                 Theme::font(13, QFont::DemiBold));

    const QRectF saveRect(1146, 38, 82, 44);
    roundedPanel(painter, saveRect, 11, Theme::primary, Theme::primary);
    drawIcon(painter, ToolIcon::Save, QPointF(1167, 60), Theme::text, 17);
    centeredText(painter, QRectF(1178, 38, 42, 44), QStringLiteral("保存"), Theme::text,
                 Theme::font(13, QFont::DemiBold));
}

void EditorCanvas::drawBottomEditor(QPainter &painter)
{
    const QRectF statusRect(150, 641, 980, 53);
    roundedPanel(painter, statusRect, 14, Theme::surface, Theme::border, 1.0);

    const int positiveCount = std::count_if(m_marks.cbegin(), m_marks.cend(), [](const SelectionMark &mark) {
        return mark.positive;
    });
    QString status;
    if (!m_transientMessage.isEmpty())
        status = m_transientMessage;
    else if (positiveCount == 0)
        status = m_captureEnabled
                     ? QStringLiteral("AI 捕捉已开启，点击图像中的微生物进行选择")
                     : QStringLiteral("AI 捕捉已关闭，可使用手动工具创建选区");
    else
        status = QStringLiteral("已捕捉 %1 个微生物，可继续调整选区").arg(positiveCount);

    drawIcon(painter, m_captureEnabled ? ToolIcon::Sparkles : ToolIcon::Eye,
             QPointF(178, 667), m_captureEnabled ? QColor(68, 126, 255) : Theme::textMuted, 17);
    centeredText(painter, QRectF(197, 647, 456, 40), status, Theme::textSecondary,
                 Theme::font(13), Qt::AlignVCenter | Qt::AlignLeft);

    const QRectF addRect(677, 646, 132, 41);
    roundedPanel(painter, addRect, 10,
                 m_addMode ? Theme::primary : Theme::surfaceSoft,
                 m_addMode ? Theme::primary : Theme::border);
    drawIcon(painter, ToolIcon::Plus, QPointF(705, 666.5), m_addMode ? Theme::text : Theme::textSecondary, 16);
    centeredText(painter, QRectF(720, 646, 78, 41), QStringLiteral("增加选区"),
                 m_addMode ? Theme::text : Theme::textSecondary, Theme::font(12, QFont::DemiBold));

    const QRectF subtractRect(817, 646, 132, 41);
    roundedPanel(painter, subtractRect, 10,
                 !m_addMode ? QColor(235, 58, 67) : Theme::surfaceSoft,
                 !m_addMode ? QColor(235, 58, 67) : Theme::border);
    drawIcon(painter, ToolIcon::Minus, QPointF(845, 666.5), !m_addMode ? Theme::text : Theme::textMuted, 16);
    centeredText(painter, QRectF(860, 646, 78, 41), QStringLiteral("减少选区"),
                 !m_addMode ? Theme::text : Theme::textMuted, Theme::font(12, QFont::DemiBold));

    const QRectF generateRect(957, 646, 164, 41);
    const bool canGenerate = positiveCount > 0;
    roundedPanel(painter, generateRect, 10,
                 canGenerate ? Theme::primary : QColor(44, 55, 75),
                 canGenerate ? Theme::primary : Theme::borderSoft);
    centeredText(painter, generateRect, QStringLiteral("生成 3D 模型"),
                 canGenerate ? Theme::text : Theme::textMuted, Theme::font(12, QFont::DemiBold));

    const QRectF leftTools(150, 706, 166, 60);
    roundedPanel(painter, leftTools, 15, Theme::surface, Theme::border);
    drawToolButton(painter, QRectF(156, 714, 48, 44), ToolIcon::Image, QStringLiteral("导入显微图像"));
    drawToolButton(painter, QRectF(208, 714, 48, 44), ToolIcon::Rotate, QStringLiteral("顺时针旋转图像"));
    drawToolButton(painter, QRectF(260, 714, 48, 44), ToolIcon::Cube, QStringLiteral("导入 OBJ 或 PLY 模型"));

    const QRectF centerTools(327, 706, 580, 60);
    roundedPanel(painter, centerTools, 15, Theme::surface, Theme::border);
    const ToolIcon centerIcons[] = {ToolIcon::Sparkles, ToolIcon::Cursor, ToolIcon::Lasso, ToolIcon::Brush,
                                    ToolIcon::Eraser, ToolIcon::Grid, ToolIcon::Eye, ToolIcon::Rotate,
                                    ToolIcon::Plus, ToolIcon::Trash, ToolIcon::Undo};
    const QString centerTips[] = {QStringLiteral("智能选择"), QStringLiteral("点选"), QStringLiteral("套索"),
                                  QStringLiteral("画笔"), QStringLiteral("擦除"), QStringLiteral("矩形选区"),
                                  QStringLiteral("显示掩膜"), QStringLiteral("椭圆选区"), QStringLiteral("文字标记"),
                                  QStringLiteral("清除全部"), QStringLiteral("撤销")};
    for (int index = 0; index < 11; ++index)
        drawToolButton(painter, QRectF(333 + index * 52, 714, 48, 44), centerIcons[index], centerTips[index],
                       index == m_activeTool);

    const QRectF aiTools(919, 706, 210, 60);
    roundedPanel(painter, aiTools, 15, Theme::surface, Theme::border);
    drawIcon(painter, ToolIcon::Sparkles, QPointF(946, 736), QColor(72, 129, 255), 18);
    centeredText(painter, QRectF(968, 716, 94, 40), QStringLiteral("AI 捕捉微生物"), Theme::textSecondary,
                 Theme::font(11, QFont::DemiBold), Qt::AlignVCenter | Qt::AlignLeft);
    const QRectF toggle(1068, 724, 47, 26);
    roundedPanel(painter, toggle, 13, m_captureEnabled ? Theme::primary : QColor(60, 73, 93));
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::white);
    painter.drawEllipse(QPointF(m_captureEnabled ? 1102 : 1081, 737), 9, 9);
}

void EditorCanvas::drawResultOverlay(QPainter &painter)
{
    const QRectF download(862, 584, 96, 41);
    drawButton(painter, download, QStringLiteral("下载3D"), false, ToolIcon::Download);
    const QRectF fullscreen(966, 584, 92, 41);
    drawButton(painter, fullscreen, QStringLiteral("全屏"), false, ToolIcon::Fullscreen);
}

void EditorCanvas::drawModal(QPainter &painter)
{
    painter.fillRect(QRectF(0, 0, 1280, 800), QColor(2, 6, 14, 166));

    if (m_state == UiState::CreditConfirm) {
        const QRectF card(390, 257, 500, 286);
        roundedPanel(painter, card, 20, Theme::surfaceRaised, Theme::border, 1.0);
        centeredText(painter, QRectF(424, 281, 382, 34), QStringLiteral("确认生成3D模型"), Theme::text,
                     Theme::font(19, QFont::DemiBold), Qt::AlignVCenter | Qt::AlignLeft);
        drawIcon(painter, ToolIcon::Close, QPointF(854, 299), Theme::textMuted, 18);
        painter.setPen(QPen(Theme::borderSoft, 1));
        painter.drawLine(QPointF(414, 328), QPointF(866, 328));

        centeredText(painter, QRectF(424, 348, 432, 28), QStringLiteral("本次转换将消耗"), Theme::textSecondary,
                     Theme::font(13), Qt::AlignVCenter | Qt::AlignLeft);
        centeredText(painter, QRectF(424, 378, 238, 45), QStringLiteral("15 积分"), Theme::text,
                     Theme::font(27, QFont::Bold), Qt::AlignVCenter | Qt::AlignLeft);
        roundedPanel(painter, QRectF(682, 374, 172, 50), 12, QColor(11, 18, 31), Theme::borderSoft);
        centeredText(painter, QRectF(696, 378, 142, 19), QStringLiteral("当前积分"), Theme::textMuted,
                     Theme::font(11), Qt::AlignVCenter | Qt::AlignLeft);
        centeredText(painter, QRectF(696, 397, 142, 22), QStringLiteral("635"), Theme::text,
                     Theme::font(16, QFont::DemiBold), Qt::AlignVCenter | Qt::AlignLeft);

        drawButton(painter, QRectF(424, 467, 206, 48), QStringLiteral("取消"), false);
        drawButton(painter, QRectF(642, 467, 214, 48), QStringLiteral("确认生成"), true, ToolIcon::Sparkles);
    } else if (m_state == UiState::Generating) {
        const QRectF card(390, 275, 500, 250);
        roundedPanel(painter, card, 20, Theme::surfaceRaised, Theme::border, 1.0);
        drawSpinner(painter, QPointF(640, 344), 26);
        centeredText(painter, QRectF(430, 387, 420, 34), QStringLiteral("正在生成3D模型"), Theme::text,
                     Theme::font(20, QFont::DemiBold));
        centeredText(painter, QRectF(430, 428, 420, 28), QStringLiteral("正在重建表面与纹理，请稍候…"), Theme::textSecondary,
                     Theme::font(13));
        roundedPanel(painter, QRectF(482, 477, 316, 6), 3, QColor(42, 55, 76));
        roundedPanel(painter, QRectF(482, 477, 86 + m_loadingPhase * 16, 6), 3, Theme::primary);
    } else if (m_state == UiState::Failed) {
        const QRectF card(390, 257, 500, 286);
        roundedPanel(painter, card, 20, Theme::surfaceRaised, Theme::border, 1.0);
        painter.setPen(QPen(QColor(255, 91, 106, 50), 10));
        painter.setBrush(QColor(255, 75, 93, 26));
        painter.drawEllipse(QPointF(640, 329), 34, 34);
        drawIcon(painter, ToolIcon::Warning, QPointF(640, 329), Theme::danger, 28);
        centeredText(painter, QRectF(424, 376, 432, 34), QStringLiteral("3D模型生成失败"), Theme::text,
                     Theme::font(20, QFont::DemiBold));
        centeredText(painter, QRectF(428, 416, 424, 44), QStringLiteral("模型数据未通过校验，积分未扣除。\n请调整选区后重试。"),
                     Theme::textSecondary, Theme::font(12), Qt::AlignCenter);
        drawButton(painter, QRectF(424, 477, 206, 48), QStringLiteral("返回编辑"), false);
        drawButton(painter, QRectF(642, 477, 214, 48), QStringLiteral("重新生成"), true, ToolIcon::Sparkles);
    }
}

void EditorCanvas::drawSavedToast(QPainter &painter)
{
    const QRectF toast(460, 116, 360, 64);
    roundedPanel(painter, toast, 14, QColor(17, 30, 43, 250), QColor(41, 133, 103));
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(20, 193, 127, 40));
    painter.drawEllipse(QPointF(493, 148), 17, 17);
    drawIcon(painter, ToolIcon::Check, QPointF(493, 148), Theme::success, 18);
    centeredText(painter, QRectF(520, 124, 270, 24), QStringLiteral("模型保存成功"), Theme::text,
                 Theme::font(14, QFont::DemiBold), Qt::AlignVCenter | Qt::AlignLeft);
    centeredText(painter, QRectF(520, 147, 270, 21), QStringLiteral("PLY 文件已保存到所选位置"), Theme::textSecondary,
                 Theme::font(11), Qt::AlignVCenter | Qt::AlignLeft);
}

void EditorCanvas::drawToolButton(QPainter &painter,
                                  const QRectF &rect,
                                  ToolIcon icon,
                                  const QString &tooltip,
                                  bool active,
                                  bool enabled)
{
    Q_UNUSED(tooltip)
    const QPointF cursor = toDesign(mapFromGlobal(QCursor::pos()));
    const bool hovered = rect.contains(cursor);
    QColor fill = Qt::transparent;
    QColor borderColor = Qt::transparent;
    if (active) {
        fill = QColor(42, 104, 255, 46);
        borderColor = QColor(70, 126, 255, 150);
    } else if (hovered) {
        fill = QColor(255, 255, 255, 13);
        borderColor = Theme::borderSoft;
    }
    if (fill.alpha() > 0)
        roundedPanel(painter, rect, 9, fill, borderColor);
    drawIcon(painter, icon, rect.center(),
             enabled ? (active ? QColor(118, 158, 255) : Theme::textSecondary) : Theme::textMuted, 19);
}

void EditorCanvas::drawButton(QPainter &painter,
                              const QRectF &rect,
                              const QString &text,
                              bool primary,
                              ToolIcon icon)
{
    const QPointF cursor = toDesign(mapFromGlobal(QCursor::pos()));
    const bool hovered = rect.contains(cursor);
    roundedPanel(painter, rect, 10,
                 primary ? (hovered ? Theme::primaryHover : Theme::primary)
                         : (hovered ? QColor(39, 51, 73) : Theme::surfaceSoft),
                 primary ? Theme::primary : Theme::border);
    const bool hasIcon = icon != ToolIcon::Cursor;
    if (hasIcon)
        drawIcon(painter, icon, QPointF(rect.center().x() - 37, rect.center().y()), Theme::text, 17);
    centeredText(painter, hasIcon ? rect.adjusted(24, 0, -4, 0) : rect, text,
                 primary ? Theme::text : Theme::textSecondary, Theme::font(13, QFont::DemiBold));
}

void EditorCanvas::drawSpinner(QPainter &painter, const QPointF &center, qreal radius)
{
    painter.save();
    painter.setPen(Qt::NoPen);
    for (int index = 0; index < 12; ++index) {
        const qreal angle = qDegreesToRadians(qreal(index * 30));
        const qreal opacity = 0.18 + 0.82 * qreal((index - m_loadingPhase + 12) % 12) / 11.0;
        QColor color = Theme::primary;
        color.setAlphaF(opacity);
        painter.setBrush(color);
        const QPointF point = center + QPointF(qCos(angle) * radius, qSin(angle) * radius);
        painter.drawEllipse(point, 3.2, 3.2);
    }
    painter.restore();
}

void EditorCanvas::drawIcon(QPainter &painter, ToolIcon icon, const QPointF &center, const QColor &color, qreal size)
{
    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);
    QPen pen(color, qMax(1.4, size / 11.0), Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    const qreal half = size * 0.5;

    switch (icon) {
    case ToolIcon::ArrowLeft:
        painter.drawLine(QPointF(center.x() + half * 0.55, center.y()), QPointF(center.x() - half * 0.55, center.y()));
        painter.drawLine(QPointF(center.x() - half * 0.55, center.y()), QPointF(center.x() - half * 0.05, center.y() - half * 0.45));
        painter.drawLine(QPointF(center.x() - half * 0.55, center.y()), QPointF(center.x() - half * 0.05, center.y() + half * 0.45));
        break;
    case ToolIcon::Close:
        painter.drawLine(center + QPointF(-half * 0.48, -half * 0.48), center + QPointF(half * 0.48, half * 0.48));
        painter.drawLine(center + QPointF(half * 0.48, -half * 0.48), center + QPointF(-half * 0.48, half * 0.48));
        break;
    case ToolIcon::Save:
        painter.drawRoundedRect(QRectF(center.x() - half * 0.68, center.y() - half * 0.68, half * 1.36, half * 1.36), 2, 2);
        painter.drawRect(QRectF(center.x() - half * 0.33, center.y() - half * 0.66, half * 0.55, half * 0.42));
        painter.drawLine(QPointF(center.x() - half * 0.35, center.y() + half * 0.23), QPointF(center.x() + half * 0.35, center.y() + half * 0.23));
        break;
    case ToolIcon::Image:
        painter.drawRoundedRect(QRectF(center.x() - half * 0.72, center.y() - half * 0.58, half * 1.44, half * 1.16), 2, 2);
        painter.drawEllipse(center + QPointF(-half * 0.30, -half * 0.20), half * 0.12, half * 0.12);
        painter.drawPolyline(QPolygonF() << center + QPointF(-half * 0.55, half * 0.35)
                                         << center + QPointF(-half * 0.12, -half * 0.02)
                                         << center + QPointF(half * 0.10, half * 0.18)
                                         << center + QPointF(half * 0.36, -half * 0.12)
                                         << center + QPointF(half * 0.58, half * 0.34));
        break;
    case ToolIcon::Rotate:
        painter.drawArc(QRectF(center.x() - half * 0.62, center.y() - half * 0.62, half * 1.24, half * 1.24), 25 * 16, 285 * 16);
        painter.drawLine(center + QPointF(half * 0.56, -half * 0.38), center + QPointF(half * 0.62, half * 0.05));
        painter.drawLine(center + QPointF(half * 0.56, -half * 0.38), center + QPointF(half * 0.16, -half * 0.27));
        break;
    case ToolIcon::Cube: {
        QPolygonF top;
        top << center + QPointF(0, -half * 0.72) << center + QPointF(half * 0.62, -half * 0.35)
            << center + QPointF(0, 0.02) << center + QPointF(-half * 0.62, -half * 0.35) << center + QPointF(0, -half * 0.72);
        painter.drawPolyline(top);
        painter.drawLine(center + QPointF(-half * 0.62, -half * 0.35), center + QPointF(-half * 0.62, half * 0.34));
        painter.drawLine(center + QPointF(half * 0.62, -half * 0.35), center + QPointF(half * 0.62, half * 0.34));
        painter.drawLine(center + QPointF(0, 0.02), center + QPointF(0, half * 0.73));
        painter.drawPolyline(QPolygonF() << center + QPointF(-half * 0.62, half * 0.34) << center + QPointF(0, half * 0.73)
                                         << center + QPointF(half * 0.62, half * 0.34));
        break;
    }
    case ToolIcon::Cursor:
        painter.drawPolyline(QPolygonF() << center + QPointF(-half * 0.52, -half * 0.66)
                                         << center + QPointF(half * 0.46, half * 0.10)
                                         << center + QPointF(half * 0.02, half * 0.14)
                                         << center + QPointF(half * 0.28, half * 0.64)
                                         << center + QPointF(0.0, half * 0.75)
                                         << center + QPointF(-half * 0.22, half * 0.24)
                                         << center + QPointF(-half * 0.52, -half * 0.66));
        break;
    case ToolIcon::Lasso:
        painter.drawEllipse(QRectF(center.x() - half * 0.65, center.y() - half * 0.48, half * 1.30, half * 0.95));
        painter.drawArc(QRectF(center.x() - half * 0.16, center.y() + half * 0.22, half * 0.62, half * 0.48), 185 * 16, 230 * 16);
        break;
    case ToolIcon::Brush:
        painter.drawLine(center + QPointF(-half * 0.45, half * 0.55), center + QPointF(half * 0.48, -half * 0.48));
        painter.drawLine(center + QPointF(half * 0.27, -half * 0.65), center + QPointF(half * 0.62, -half * 0.30));
        painter.drawArc(QRectF(center.x() - half * 0.70, center.y() + half * 0.22, half * 0.72, half * 0.48), 180 * 16, 180 * 16);
        break;
    case ToolIcon::Eraser:
        painter.drawRoundedRect(QRectF(center.x() - half * 0.55, center.y() - half * 0.38, half * 1.10, half * 0.76), 2, 2);
        painter.drawLine(center + QPointF(0, -half * 0.38), center + QPointF(0, half * 0.38));
        break;
    case ToolIcon::Sparkles:
        painter.drawLine(center + QPointF(0, -half * 0.72), center + QPointF(0, half * 0.72));
        painter.drawLine(center + QPointF(-half * 0.72, 0), center + QPointF(half * 0.72, 0));
        painter.drawLine(center + QPointF(-half * 0.40, -half * 0.40), center + QPointF(half * 0.40, half * 0.40));
        painter.drawLine(center + QPointF(half * 0.40, -half * 0.40), center + QPointF(-half * 0.40, half * 0.40));
        break;
    case ToolIcon::Eye:
        painter.drawPath(QPainterPath(center + QPointF(-half * 0.72, 0)));
        painter.drawEllipse(center, half * 0.26, half * 0.26);
        painter.drawArc(QRectF(center.x() - half * 0.75, center.y() - half * 0.50, half * 1.5, half), 0, 180 * 16);
        painter.drawArc(QRectF(center.x() - half * 0.75, center.y(), half * 1.5, half), 180 * 16, 180 * 16);
        break;
    case ToolIcon::Grid:
        painter.drawRect(QRectF(center.x() - half * 0.62, center.y() - half * 0.62, half * 1.24, half * 1.24));
        painter.drawLine(QPointF(center.x(), center.y() - half * 0.62), QPointF(center.x(), center.y() + half * 0.62));
        painter.drawLine(QPointF(center.x() - half * 0.62, center.y()), QPointF(center.x() + half * 0.62, center.y()));
        break;
    case ToolIcon::Plus:
        painter.drawLine(QPointF(center.x() - half * 0.5, center.y()), QPointF(center.x() + half * 0.5, center.y()));
        painter.drawLine(QPointF(center.x(), center.y() - half * 0.5), QPointF(center.x(), center.y() + half * 0.5));
        break;
    case ToolIcon::Minus:
        painter.drawLine(QPointF(center.x() - half * 0.5, center.y()), QPointF(center.x() + half * 0.5, center.y()));
        break;
    case ToolIcon::Undo:
        painter.drawArc(QRectF(center.x() - half * 0.55, center.y() - half * 0.48, half * 1.15, half * 1.10), -55 * 16, 250 * 16);
        painter.drawLine(center + QPointF(-half * 0.50, -half * 0.10), center + QPointF(-half * 0.65, -half * 0.50));
        painter.drawLine(center + QPointF(-half * 0.50, -half * 0.10), center + QPointF(-half * 0.12, -half * 0.30));
        break;
    case ToolIcon::Trash:
        painter.drawRoundedRect(QRectF(center.x() - half * 0.43, center.y() - half * 0.42, half * 0.86, half * 1.03), 2, 2);
        painter.drawLine(QPointF(center.x() - half * 0.58, center.y() - half * 0.55), QPointF(center.x() + half * 0.58, center.y() - half * 0.55));
        painter.drawLine(QPointF(center.x() - half * 0.20, center.y() - half * 0.74), QPointF(center.x() + half * 0.20, center.y() - half * 0.74));
        break;
    case ToolIcon::Download:
        painter.drawLine(QPointF(center.x(), center.y() - half * 0.68), QPointF(center.x(), center.y() + half * 0.20));
        painter.drawLine(center + QPointF(-half * 0.34, -half * 0.08), center + QPointF(0, half * 0.27));
        painter.drawLine(center + QPointF(half * 0.34, -half * 0.08), center + QPointF(0, half * 0.27));
        painter.drawLine(QPointF(center.x() - half * 0.58, center.y() + half * 0.62), QPointF(center.x() + half * 0.58, center.y() + half * 0.62));
        break;
    case ToolIcon::Fullscreen:
        painter.drawLine(center + QPointF(-half * 0.62, -half * 0.10), center + QPointF(-half * 0.62, -half * 0.62));
        painter.drawLine(center + QPointF(-half * 0.62, -half * 0.62), center + QPointF(-half * 0.10, -half * 0.62));
        painter.drawLine(center + QPointF(half * 0.62, -half * 0.10), center + QPointF(half * 0.62, -half * 0.62));
        painter.drawLine(center + QPointF(half * 0.62, -half * 0.62), center + QPointF(half * 0.10, -half * 0.62));
        painter.drawLine(center + QPointF(-half * 0.62, half * 0.10), center + QPointF(-half * 0.62, half * 0.62));
        painter.drawLine(center + QPointF(-half * 0.62, half * 0.62), center + QPointF(-half * 0.10, half * 0.62));
        painter.drawLine(center + QPointF(half * 0.62, half * 0.10), center + QPointF(half * 0.62, half * 0.62));
        painter.drawLine(center + QPointF(half * 0.62, half * 0.62), center + QPointF(half * 0.10, half * 0.62));
        break;
    case ToolIcon::Home:
        painter.drawPolyline(QPolygonF() << center + QPointF(-half * 0.62, -half * 0.02)
                                         << center + QPointF(0, -half * 0.62)
                                         << center + QPointF(half * 0.62, -half * 0.02));
        painter.drawRect(QRectF(center.x() - half * 0.45, center.y() - half * 0.02, half * 0.9, half * 0.65));
        break;
    case ToolIcon::Check:
        painter.drawPolyline(QPolygonF() << center + QPointF(-half * 0.56, 0)
                                         << center + QPointF(-half * 0.12, half * 0.42)
                                         << center + QPointF(half * 0.62, -half * 0.48));
        break;
    case ToolIcon::Warning:
        painter.drawPolygon(QPolygonF() << center + QPointF(0, -half * 0.72)
                                        << center + QPointF(half * 0.70, half * 0.60)
                                        << center + QPointF(-half * 0.70, half * 0.60));
        painter.drawLine(QPointF(center.x(), center.y() - half * 0.28), QPointF(center.x(), center.y() + half * 0.18));
        painter.drawPoint(QPointF(center.x(), center.y() + half * 0.40));
        break;
    }
    painter.restore();
}

void EditorCanvas::addSelection(const QPointF &position, bool positive)
{
    SelectionMark mark;
    mark.center = position;
    mark.positive = positive;
    mark.radiusX = 72.0 + (m_marks.size() % 3) * 10.0;
    mark.radiusY = 46.0 + (m_marks.size() % 2) * 12.0;
    m_marks.append(mark);
    m_state = UiState::Selected;
    if (m_usingBundledSample)
        m_addMode = false;
    update();
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
        showTransientMessage(QStringLiteral("无法读取所选图像"));
        return;
    }
    m_sourceImage = image;
    m_usingBundledSample = false;
    m_imageName = QFileInfo(fileName).completeBaseName();
    m_marks.clear();
    m_state = UiState::Waiting;
    showTransientMessage(QStringLiteral("图像已导入"));
    update();
}

void EditorCanvas::rotateImage()
{
    if (m_sourceImage.isNull())
        return;
    QTransform transform;
    transform.rotate(90.0);
    m_sourceImage = m_sourceImage.transformed(transform, Qt::SmoothTransformation);
    update();
}

void EditorCanvas::openModel()
{
    const QString initial = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    const QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("导入3D模型"), initial,
                                                           QStringLiteral("3D 模型 (*.ply *.obj)"));
    if (fileName.isEmpty())
        return;
    QString error;
    if (!m_model.load(fileName, &error)) {
        m_state = UiState::Failed;
        showTransientMessage(error);
        update();
        return;
    }
    m_modelName = QFileInfo(fileName).completeBaseName();
    resetModelView();
    m_state = UiState::Result;
    update();
}

void EditorCanvas::saveModel()
{
    if (m_model.isEmpty()) {
        showTransientMessage(QStringLiteral("没有可保存的3D模型"));
        return;
    }
    const QString initial = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                            + QStringLiteral("/sam-3d-model.ply");
    const QString fileName = QFileDialog::getSaveFileName(this, QStringLiteral("保存3D模型"), initial,
                                                           QStringLiteral("PLY 模型 (*.ply)"));
    if (fileName.isEmpty())
        return;
    QString output = fileName;
    if (!output.endsWith(QStringLiteral(".ply"), Qt::CaseInsensitive))
        output += QStringLiteral(".ply");
    QString error;
    if (!m_model.savePly(output, &error)) {
        showTransientMessage(error);
        return;
    }
    m_savedToastVisible = true;
    m_toastTimer.start(3200);
    update();
}

void EditorCanvas::beginGeneration()
{
    m_state = UiState::Generating;
    m_loadingPhase = 0;
    m_generationTimer.start(1550);
    update();
}

void EditorCanvas::completeGeneration()
{
    if (m_demoStateLocked)
        return;
    if (m_model.isEmpty())
        m_model.createOrganicSample();
    m_modelName = QStringLiteral("叶片表皮 · 微生物重建");
    resetModelView();
    m_state = UiState::Result;
    update();
}

void EditorCanvas::resetModelView()
{
    m_rotationX = -14.0f;
    m_rotationY = 26.0f;
    m_zoom = 0.82f;
    m_pan = QPointF();
    update();
}

void EditorCanvas::toggleFullscreen()
{
    if (isFullScreen())
        showNormal();
    else
        showFullScreen();
    update();
}

void EditorCanvas::showTransientMessage(const QString &message)
{
    m_transientMessage = message;
    m_messageTimer.start(2600);
    update();
}

void EditorCanvas::handleClick(const QPointF &position)
{
    if (m_state == UiState::CreditConfirm) {
        if (QRectF(562, 472, 142, 48).contains(position)) {
            m_state = UiState::Selected;
            update();
        } else if (QRectF(716, 472, 142, 48).contains(position)) {
            m_demoStateLocked = false;
            beginGeneration();
        }
        return;
    }
    if (m_state == UiState::Generating)
        return;
    if (m_state == UiState::Failed) {
        if (QRectF(514, 459, 120, 44).contains(position)) {
            m_state = m_marks.isEmpty() ? UiState::Waiting : UiState::Selected;
            update();
        } else if (QRectF(646, 459, 120, 44).contains(position)) {
            m_demoStateLocked = false;
            beginGeneration();
        }
        return;
    }

    if (QRectF(52, 38, 92, 44).contains(position)) {
        if (isResultState())
            m_state = m_marks.isEmpty() ? UiState::Waiting : UiState::Selected;
        else
            close();
        update();
        return;
    }
    if (QRectF(638, 39, 144, 42).contains(position)) {
        close();
        return;
    }
    if (QRectF(1146, 38, 82, 44).contains(position)) {
        if (isResultState())
            saveModel();
        else
            showTransientMessage(QStringLiteral("请先生成3D模型"));
        return;
    }

    if (isResultState()) {
        if (QRectF(862, 584, 96, 41).contains(position))
            saveModel();
        else if (QRectF(966, 584, 92, 41).contains(position))
            toggleFullscreen();
        return;
    }

    if (QRectF(156, 714, 48, 44).contains(position)) {
        openImage();
        return;
    }
    if (QRectF(208, 714, 48, 44).contains(position)) {
        rotateImage();
        return;
    }
    if (QRectF(260, 714, 48, 44).contains(position)) {
        openModel();
        return;
    }
    if (QRectF(1068, 724, 47, 26).contains(position)) {
        m_captureEnabled = !m_captureEnabled;
        update();
        return;
    }
    if (QRectF(677, 646, 132, 41).contains(position)) {
        m_addMode = true;
        update();
        return;
    }
    if (QRectF(817, 646, 132, 41).contains(position)) {
        m_addMode = false;
        update();
        return;
    }
    if (QRectF(957, 646, 164, 41).contains(position)) {
        if (!m_marks.isEmpty()) {
            m_state = UiState::CreditConfirm;
            update();
        }
        return;
    }
    for (int index = 0; index < 11; ++index) {
        if (!QRectF(333 + index * 52, 714, 48, 44).contains(position))
            continue;
        if (index == 10 && !m_marks.isEmpty()) {
            m_marks.removeLast();
            m_state = m_marks.isEmpty() ? UiState::Waiting : UiState::Selected;
        } else if (index == 9) {
            m_marks.clear();
            m_state = UiState::Waiting;
        } else {
            m_activeTool = index;
        }
        update();
        return;
    }

    if (position.y() > 108 && position.y() < 626)
        addSelection(position, m_addMode);
}

QString EditorCanvas::tooltipAt(const QPointF &position) const
{
    if (QRectF(156, 714, 48, 44).contains(position)) return QStringLiteral("导入显微图像 (O)");
    if (QRectF(208, 714, 48, 44).contains(position)) return QStringLiteral("顺时针旋转图像");
    if (QRectF(260, 714, 48, 44).contains(position)) return QStringLiteral("导入 OBJ 或 PLY 模型 (M)");
    if (QRectF(862, 584, 96, 41).contains(position)) return QStringLiteral("保存 PLY 模型 (Ctrl+S)");
    if (QRectF(966, 584, 92, 41).contains(position)) return QStringLiteral("切换全屏 (F11)");
    if (QRectF(677, 646, 132, 41).contains(position)) return QStringLiteral("增加选区");
    if (QRectF(817, 646, 132, 41).contains(position)) return QStringLiteral("减少选区");
    return QString();
}

void EditorCanvas::mousePressEvent(QMouseEvent *event)
{
    setFocus(Qt::MouseFocusReason);
    m_lastMouse = event->localPos();
    m_pressDesign = toDesign(event->localPos());
    m_mouseMoved = false;

    if (event->button() == Qt::LeftButton && QRectF(160, 28, 460, 64).contains(m_pressDesign)) {
        m_draggingWindow = true;
        m_windowDragOffset = event->globalPos() - frameGeometry().topLeft();
        event->accept();
        return;
    }

    if (isResultState() && m_pressDesign.x() >= 640 && m_pressDesign.y() > 98 && m_pressDesign.y() < 636) {
        if (event->button() == Qt::LeftButton)
            m_rotatingModel = true;
        else if (event->button() == Qt::RightButton || event->button() == Qt::MiddleButton)
            m_panningModel = true;
    }
    event->accept();
}

void EditorCanvas::mouseMoveEvent(QMouseEvent *event)
{
    const QPointF delta = event->localPos() - m_lastMouse;
    if (QLineF(event->localPos(), QPointF(m_pressDesign.x() * width() / kDesignWidth,
                                          m_pressDesign.y() * height() / kDesignHeight)).length() > 3.0)
        m_mouseMoved = true;

    if (m_draggingWindow) {
        if (!isMaximized() && !isFullScreen())
            move(event->globalPos() - m_windowDragOffset);
    } else if (m_rotatingModel) {
        m_rotationY += float(delta.x() * 0.55);
        m_rotationX += float(delta.y() * 0.55);
        m_rotationX = qBound(-89.0f, m_rotationX, 89.0f);
        update();
    } else if (m_panningModel) {
        m_pan += delta * (kDesignWidth / qMax(1, width()));
        update();
    } else {
        const QString tooltip = tooltipAt(toDesign(event->localPos()));
        if (!tooltip.isEmpty())
            QToolTip::showText(event->globalPos(), tooltip, this);
        else
            QToolTip::hideText();
        update();
    }
    m_lastMouse = event->localPos();
    event->accept();
}

void EditorCanvas::mouseReleaseEvent(QMouseEvent *event)
{
    const bool clicked = !m_mouseMoved;
    m_rotatingModel = false;
    m_panningModel = false;
    m_draggingWindow = false;
    if (clicked && event->button() == Qt::LeftButton)
        handleClick(toDesign(event->localPos()));
    event->accept();
}

void EditorCanvas::mouseDoubleClickEvent(QMouseEvent *event)
{
    const QPointF position = toDesign(event->localPos());
    if (isResultState() && position.x() >= 640) {
        resetModelView();
    } else if (QRectF(160, 28, 460, 64).contains(position)) {
        isMaximized() ? showNormal() : showMaximized();
    }
    event->accept();
}

void EditorCanvas::wheelEvent(QWheelEvent *event)
{
    const QPointF position = toDesign(event->posF());
    if (isResultState() && position.x() >= 640) {
        const float factor = event->angleDelta().y() > 0 ? 1.11f : 0.90f;
        m_zoom = qBound(0.35f, m_zoom * factor, 4.5f);
        update();
        event->accept();
        return;
    }
    QOpenGLWidget::wheelEvent(event);
}

void EditorCanvas::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Escape) {
        if (isFullScreen())
            showNormal();
        else if (m_state == UiState::CreditConfirm || m_state == UiState::Failed)
            m_state = m_marks.isEmpty() ? UiState::Waiting : UiState::Selected;
        else if (m_state == UiState::Result)
            m_state = m_marks.isEmpty() ? UiState::Waiting : UiState::Selected;
        update();
        return;
    }
    if (event->matches(QKeySequence::Open) || event->key() == Qt::Key_O) {
        openImage();
        return;
    }
    if (event->key() == Qt::Key_M) {
        openModel();
        return;
    }
    if (event->matches(QKeySequence::Save)) {
        saveModel();
        return;
    }
    if (event->key() == Qt::Key_F11) {
        toggleFullscreen();
        return;
    }
    if (isResultState()) {
        if (event->key() == Qt::Key_Left) m_rotationY -= 5.0f;
        else if (event->key() == Qt::Key_Right) m_rotationY += 5.0f;
        else if (event->key() == Qt::Key_Up) m_rotationX -= 5.0f;
        else if (event->key() == Qt::Key_Down) m_rotationX += 5.0f;
        else if (event->key() == Qt::Key_Plus || event->key() == Qt::Key_Equal) m_zoom = qMin(4.5f, m_zoom * 1.1f);
        else if (event->key() == Qt::Key_Minus) m_zoom = qMax(0.35f, m_zoom * 0.9f);
        else {
            QOpenGLWidget::keyPressEvent(event);
            return;
        }
        update();
        return;
    }
    QOpenGLWidget::keyPressEvent(event);
}

void EditorCanvas::leaveEvent(QEvent *event)
{
    QToolTip::hideText();
    QOpenGLWidget::leaveEvent(event);
}

void EditorCanvas::setDemoState(const QString &stateName)
{
    const QString name = stateName.trimmed().toLower();
    m_demoStateLocked = true;
    m_marks.clear();
    m_savedToastVisible = false;

    if (name != QStringLiteral("waiting")) {
        addSelection(QPointF(478, 344), true);
        addSelection(QPointF(765, 426), true);
        addSelection(QPointF(927, 289), true);
    }

    if (name == QStringLiteral("confirm"))
        m_state = UiState::CreditConfirm;
    else if (name == QStringLiteral("generating"))
        m_state = UiState::Generating;
    else if (name == QStringLiteral("failed"))
        m_state = UiState::Failed;
    else if (name == QStringLiteral("result") || name == QStringLiteral("saved")) {
        m_state = UiState::Result;
        if (m_model.isEmpty())
            m_model.createOrganicSample();
        m_modelName = QStringLiteral("叶片表皮 · 微生物重建");
        if (name == QStringLiteral("saved"))
            m_savedToastVisible = true;
    } else if (name == QStringLiteral("selected")) {
        m_state = UiState::Selected;
    } else {
        m_marks.clear();
        m_state = UiState::Waiting;
    }
    update();
}
