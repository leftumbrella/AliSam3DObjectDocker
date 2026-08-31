#include "modeldata.h"

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QSaveFile>
#include <QTextStream>
#include <QtEndian>
#include <QtMath>

#include <cstring>
#include <cctype>
#include <limits>

namespace {

enum class PlyType {
    Invalid,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64
};

struct PlyProperty {
    QString name;
    PlyType type = PlyType::Invalid;
    bool list = false;
    PlyType countType = PlyType::Invalid;
    PlyType valueType = PlyType::Invalid;
};

PlyType plyType(const QByteArray &value)
{
    const QByteArray type = value.toLower();
    if (type == "char" || type == "int8") return PlyType::Int8;
    if (type == "uchar" || type == "uint8") return PlyType::UInt8;
    if (type == "short" || type == "int16") return PlyType::Int16;
    if (type == "ushort" || type == "uint16") return PlyType::UInt16;
    if (type == "int" || type == "int32") return PlyType::Int32;
    if (type == "uint" || type == "uint32") return PlyType::UInt32;
    if (type == "float" || type == "float32") return PlyType::Float32;
    if (type == "double" || type == "float64") return PlyType::Float64;
    return PlyType::Invalid;
}

int plyTypeSize(PlyType type)
{
    switch (type) {
    case PlyType::Int8:
    case PlyType::UInt8: return 1;
    case PlyType::Int16:
    case PlyType::UInt16: return 2;
    case PlyType::Int32:
    case PlyType::UInt32:
    case PlyType::Float32: return 4;
    case PlyType::Float64: return 8;
    default: return 0;
    }
}

class AsciiCursor
{
public:
    explicit AsciiCursor(const QByteArray &data)
        : m_current(data.constData()), m_end(data.constData() + data.size())
    {
    }

    QByteArray next()
    {
        while (m_current < m_end && std::isspace(static_cast<unsigned char>(*m_current)))
            ++m_current;
        const char *start = m_current;
        while (m_current < m_end && !std::isspace(static_cast<unsigned char>(*m_current)))
            ++m_current;
        return QByteArray(start, int(m_current - start));
    }

private:
    const char *m_current;
    const char *m_end;
};

bool readBinaryScalar(const char *&cursor, const char *end, PlyType type, double &value)
{
    const int size = plyTypeSize(type);
    if (size <= 0 || cursor + size > end)
        return false;

    switch (type) {
    case PlyType::Int8:
        value = *reinterpret_cast<const qint8 *>(cursor);
        break;
    case PlyType::UInt8:
        value = *reinterpret_cast<const quint8 *>(cursor);
        break;
    case PlyType::Int16: {
        qint16 raw;
        std::memcpy(&raw, cursor, sizeof(raw));
        value = qFromLittleEndian(raw);
        break;
    }
    case PlyType::UInt16: {
        quint16 raw;
        std::memcpy(&raw, cursor, sizeof(raw));
        value = qFromLittleEndian(raw);
        break;
    }
    case PlyType::Int32: {
        qint32 raw;
        std::memcpy(&raw, cursor, sizeof(raw));
        value = qFromLittleEndian(raw);
        break;
    }
    case PlyType::UInt32: {
        quint32 raw;
        std::memcpy(&raw, cursor, sizeof(raw));
        value = qFromLittleEndian(raw);
        break;
    }
    case PlyType::Float32: {
        quint32 bits;
        std::memcpy(&bits, cursor, sizeof(bits));
        bits = qFromLittleEndian(bits);
        float raw;
        std::memcpy(&raw, &bits, sizeof(raw));
        value = raw;
        break;
    }
    case PlyType::Float64: {
        quint64 bits;
        std::memcpy(&bits, cursor, sizeof(bits));
        bits = qFromLittleEndian(bits);
        double raw;
        std::memcpy(&raw, &bits, sizeof(raw));
        value = raw;
        break;
    }
    default:
        return false;
    }

    cursor += size;
    return true;
}

QColor colorFromValues(double red, double green, double blue)
{
    const double maxValue = qMax(red, qMax(green, blue));
    if (maxValue <= 1.0)
        return QColor::fromRgbF(qBound(0.0, red, 1.0), qBound(0.0, green, 1.0), qBound(0.0, blue, 1.0));
    return QColor(qBound(0, qRound(red), 255), qBound(0, qRound(green), 255), qBound(0, qRound(blue), 255));
}

void setError(QString *error, const QString &message)
{
    if (error)
        *error = message;
}

} // namespace

void ModelData::clear()
{
    vertices.clear();
    normals.clear();
    colors.clear();
    indices.clear();
}

bool ModelData::isEmpty() const
{
    return vertices.isEmpty();
}

int ModelData::triangleCount() const
{
    return indices.size() / 3;
}

bool ModelData::load(const QString &fileName, QString *error)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        setError(error, QStringLiteral("无法打开模型文件：%1").arg(file.errorString()));
        return false;
    }

    const QByteArray data = file.readAll();
    if (data.isEmpty()) {
        setError(error, QStringLiteral("模型文件为空"));
        return false;
    }

    clear();
    const QString suffix = QFileInfo(fileName).suffix().toLower();
    bool loaded = false;
    if (suffix == QStringLiteral("ply") || data.startsWith("ply"))
        loaded = loadPly(data, error);
    else if (suffix == QStringLiteral("obj"))
        loaded = loadObj(data, error);
    else
        setError(error, QStringLiteral("仅支持 OBJ、ASCII PLY 和二进制小端 PLY"));

    if (!loaded) {
        clear();
        return false;
    }

    normalizeGeometry();
    calculateNormals();
    ensureColors();
    return true;
}

bool ModelData::loadObj(const QByteArray &data, QString *error)
{
    const QList<QByteArray> lines = data.split('\n');
    for (const QByteArray &rawLine : lines) {
        const QByteArray line = rawLine.trimmed();
        if (line.isEmpty() || line.startsWith('#'))
            continue;

        const QList<QByteArray> values = line.simplified().split(' ');
        if (values.isEmpty())
            continue;

        if (values.first() == "v" && values.size() >= 4) {
            vertices.append(QVector3D(values.at(1).toFloat(), values.at(2).toFloat(), values.at(3).toFloat()));
            if (values.size() >= 7)
                colors.append(colorFromValues(values.at(4).toDouble(), values.at(5).toDouble(), values.at(6).toDouble()));
            else
                colors.append(QColor());
        } else if (values.first() == "f" && values.size() >= 4) {
            QVector<quint32> face;
            for (int i = 1; i < values.size(); ++i) {
                const QByteArray indexToken = values.at(i).split('/').value(0);
                bool ok = false;
                int index = indexToken.toInt(&ok);
                if (!ok || index == 0)
                    continue;
                if (index < 0)
                    index = vertices.size() + index;
                else
                    index -= 1;
                if (index >= 0 && index < vertices.size())
                    face.append(quint32(index));
            }
            for (int i = 1; i + 1 < face.size(); ++i)
                indices << face.at(0) << face.at(i) << face.at(i + 1);
        }
    }

    if (vertices.isEmpty()) {
        setError(error, QStringLiteral("OBJ 中没有可用顶点"));
        return false;
    }
    return true;
}

bool ModelData::loadPly(const QByteArray &data, QString *error)
{
    const int endHeader = data.indexOf("end_header");
    if (endHeader < 0) {
        setError(error, QStringLiteral("PLY 缺少 end_header"));
        return false;
    }
    int bodyOffset = data.indexOf('\n', endHeader);
    if (bodyOffset < 0) {
        setError(error, QStringLiteral("PLY 头部不完整"));
        return false;
    }
    ++bodyOffset;

    const QList<QByteArray> headerLines = data.left(bodyOffset).split('\n');
    bool ascii = false;
    bool binaryLittleEndian = false;
    int vertexCount = 0;
    int faceCount = 0;
    QByteArray element;
    QVector<PlyProperty> vertexProperties;
    QVector<PlyProperty> faceProperties;

    for (const QByteArray &rawLine : headerLines) {
        const QList<QByteArray> values = rawLine.trimmed().simplified().split(' ');
        if (values.isEmpty())
            continue;
        if (values.value(0) == "format" && values.size() >= 2) {
            ascii = values.at(1) == "ascii";
            binaryLittleEndian = values.at(1) == "binary_little_endian";
        } else if (values.value(0) == "element" && values.size() >= 3) {
            element = values.at(1);
            if (element == "vertex")
                vertexCount = values.at(2).toInt();
            else if (element == "face")
                faceCount = values.at(2).toInt();
        } else if (values.value(0) == "property") {
            PlyProperty property;
            if (values.value(1) == "list" && values.size() >= 5) {
                property.list = true;
                property.countType = plyType(values.at(2));
                property.valueType = plyType(values.at(3));
                property.name = QString::fromLatin1(values.at(4));
            } else if (values.size() >= 3) {
                property.type = plyType(values.at(1));
                property.name = QString::fromLatin1(values.at(2));
            }
            if (element == "vertex")
                vertexProperties.append(property);
            else if (element == "face")
                faceProperties.append(property);
        }
    }

    if ((!ascii && !binaryLittleEndian) || vertexCount <= 0 || vertexProperties.isEmpty()) {
        setError(error, QStringLiteral("PLY 格式不受支持或没有顶点"));
        return false;
    }

    vertices.reserve(vertexCount);
    colors.reserve(vertexCount);
    normals.reserve(vertexCount);

    auto appendVertex = [this](const QHash<QString, double> &value) {
        vertices.append(QVector3D(float(value.value(QStringLiteral("x"))),
                                  float(value.value(QStringLiteral("y"))),
                                  float(value.value(QStringLiteral("z")))));
        const bool hasNormal = value.contains(QStringLiteral("nx")) &&
                               value.contains(QStringLiteral("ny")) &&
                               value.contains(QStringLiteral("nz"));
        normals.append(hasNormal
                           ? QVector3D(float(value.value(QStringLiteral("nx"))),
                                       float(value.value(QStringLiteral("ny"))),
                                       float(value.value(QStringLiteral("nz"))))
                           : QVector3D());
        const bool hasColor = value.contains(QStringLiteral("red")) &&
                              value.contains(QStringLiteral("green")) &&
                              value.contains(QStringLiteral("blue"));
        colors.append(hasColor
                          ? colorFromValues(value.value(QStringLiteral("red")),
                                            value.value(QStringLiteral("green")),
                                            value.value(QStringLiteral("blue")))
                          : QColor());
    };

    if (ascii) {
        const QByteArray body = data.mid(bodyOffset);
        AsciiCursor cursor(body);
        for (int vertex = 0; vertex < vertexCount; ++vertex) {
            QHash<QString, double> values;
            for (const PlyProperty &property : vertexProperties) {
                const QByteArray token = cursor.next();
                if (token.isEmpty()) {
                    setError(error, QStringLiteral("PLY 顶点数据提前结束"));
                    return false;
                }
                values.insert(property.name, token.toDouble());
            }
            appendVertex(values);
        }

        for (int face = 0; face < faceCount; ++face) {
            for (const PlyProperty &property : faceProperties) {
                if (property.list) {
                    const int count = cursor.next().toInt();
                    QVector<quint32> faceIndices;
                    faceIndices.reserve(count);
                    for (int i = 0; i < count; ++i)
                        faceIndices.append(cursor.next().toUInt());
                    if (property.name == QStringLiteral("vertex_indices") ||
                        property.name == QStringLiteral("vertex_index")) {
                        for (int i = 1; i + 1 < faceIndices.size(); ++i)
                            indices << faceIndices.at(0) << faceIndices.at(i) << faceIndices.at(i + 1);
                    }
                } else {
                    cursor.next();
                }
            }
        }
    } else {
        const char *cursor = data.constData() + bodyOffset;
        const char *end = data.constData() + data.size();
        for (int vertex = 0; vertex < vertexCount; ++vertex) {
            QHash<QString, double> values;
            for (const PlyProperty &property : vertexProperties) {
                double value = 0.0;
                if (!readBinaryScalar(cursor, end, property.type, value)) {
                    setError(error, QStringLiteral("二进制 PLY 顶点数据提前结束"));
                    return false;
                }
                values.insert(property.name, value);
            }
            appendVertex(values);
        }

        for (int face = 0; face < faceCount; ++face) {
            for (const PlyProperty &property : faceProperties) {
                if (property.list) {
                    double countValue = 0.0;
                    if (!readBinaryScalar(cursor, end, property.countType, countValue)) {
                        setError(error, QStringLiteral("二进制 PLY 面数据提前结束"));
                        return false;
                    }
                    const int count = qMax(0, int(countValue));
                    QVector<quint32> faceIndices;
                    faceIndices.reserve(count);
                    for (int i = 0; i < count; ++i) {
                        double indexValue = 0.0;
                        if (!readBinaryScalar(cursor, end, property.valueType, indexValue)) {
                            setError(error, QStringLiteral("二进制 PLY 索引数据提前结束"));
                            return false;
                        }
                        faceIndices.append(quint32(qMax(0, int(indexValue))));
                    }
                    if (property.name == QStringLiteral("vertex_indices") ||
                        property.name == QStringLiteral("vertex_index")) {
                        for (int i = 1; i + 1 < faceIndices.size(); ++i)
                            indices << faceIndices.at(0) << faceIndices.at(i) << faceIndices.at(i + 1);
                    }
                } else {
                    double ignored = 0.0;
                    if (!readBinaryScalar(cursor, end, property.type, ignored)) {
                        setError(error, QStringLiteral("二进制 PLY 面属性提前结束"));
                        return false;
                    }
                }
            }
        }
    }

    for (int i = indices.size() - 1; i >= 0; --i) {
        if (indices.at(i) >= quint32(vertices.size())) {
            setError(error, QStringLiteral("PLY 面索引越界"));
            return false;
        }
    }
    return !vertices.isEmpty();
}

void ModelData::normalizeGeometry()
{
    if (vertices.isEmpty())
        return;

    QVector3D minimum(std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max());
    QVector3D maximum(-std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max());

    for (const QVector3D &vertex : qAsConst(vertices)) {
        minimum.setX(qMin(minimum.x(), vertex.x()));
        minimum.setY(qMin(minimum.y(), vertex.y()));
        minimum.setZ(qMin(minimum.z(), vertex.z()));
        maximum.setX(qMax(maximum.x(), vertex.x()));
        maximum.setY(qMax(maximum.y(), vertex.y()));
        maximum.setZ(qMax(maximum.z(), vertex.z()));
    }

    const QVector3D center = (minimum + maximum) * 0.5f;
    const QVector3D size = maximum - minimum;
    const float largest = qMax(size.x(), qMax(size.y(), size.z()));
    const float scale = largest > 0.000001f ? 2.0f / largest : 1.0f;
    for (QVector3D &vertex : vertices)
        vertex = (vertex - center) * scale;
}

void ModelData::calculateNormals()
{
    normals.fill(QVector3D(), vertices.size());
    for (int i = 0; i + 2 < indices.size(); i += 3) {
        const int a = int(indices.at(i));
        const int b = int(indices.at(i + 1));
        const int c = int(indices.at(i + 2));
        if (a < 0 || b < 0 || c < 0 || a >= vertices.size() || b >= vertices.size() || c >= vertices.size())
            continue;
        const QVector3D normal = QVector3D::crossProduct(vertices.at(b) - vertices.at(a),
                                                          vertices.at(c) - vertices.at(a));
        normals[a] += normal;
        normals[b] += normal;
        normals[c] += normal;
    }
    for (int i = 0; i < normals.size(); ++i) {
        if (normals.at(i).lengthSquared() < 0.000001f)
            normals[i] = vertices.at(i).normalized();
        else
            normals[i].normalize();
    }
}

void ModelData::ensureColors()
{
    colors.resize(vertices.size());
    for (int i = 0; i < colors.size(); ++i) {
        if (colors.at(i).isValid())
            continue;
        const float depth = qBound(0.0f, (vertices.at(i).z() + 1.0f) * 0.5f, 1.0f);
        colors[i] = QColor::fromRgbF(0.045f + depth * 0.045f,
                                     0.58f + depth * 0.24f,
                                     0.38f + depth * 0.22f);
    }
}

void ModelData::createOrganicSample()
{
    clear();

    auto appendEllipsoid = [this](const QVector3D &center,
                                  const QVector3D &radius,
                                  int latitudeSegments,
                                  int longitudeSegments,
                                  float phase) {
        const int base = vertices.size();
        for (int latitude = 0; latitude <= latitudeSegments; ++latitude) {
            const float v = float(latitude) / float(latitudeSegments);
            const float theta = float(M_PI) * v;
            for (int longitude = 0; longitude <= longitudeSegments; ++longitude) {
                const float u = float(longitude) / float(longitudeSegments);
                const float phi = float(M_PI * 2.0) * u;
                const float texture = 1.0f + 0.055f * qSin(phi * 7.0f + theta * 5.0f + phase);
                QVector3D point(qSin(theta) * qCos(phi) * radius.x(),
                                qCos(theta) * radius.y(),
                                qSin(theta) * qSin(phi) * radius.z());
                point *= texture;
                vertices.append(center + point);
                colors.append(QColor());
            }
        }
        for (int latitude = 0; latitude < latitudeSegments; ++latitude) {
            for (int longitude = 0; longitude < longitudeSegments; ++longitude) {
                const quint32 a = quint32(base + latitude * (longitudeSegments + 1) + longitude);
                const quint32 b = a + quint32(longitudeSegments + 1);
                indices << a << b << a + 1;
                indices << a + 1 << b << b + 1;
            }
        }
    };

    auto appendCurvedTube = [this](const QVector3D &start,
                                   const QVector3D &control,
                                   const QVector3D &end,
                                   float radius,
                                   int pathSegments,
                                   int sides) {
        const int base = vertices.size();
        for (int segment = 0; segment <= pathSegments; ++segment) {
            const float t = float(segment) / float(pathSegments);
            const float inverse = 1.0f - t;
            const QVector3D center = inverse * inverse * start + 2.0f * inverse * t * control + t * t * end;
            const float nextT = qMin(1.0f, t + 0.01f);
            const float nextInverse = 1.0f - nextT;
            const QVector3D next = nextInverse * nextInverse * start + 2.0f * nextInverse * nextT * control + nextT * nextT * end;
            QVector3D tangent = (next - center).normalized();
            if (tangent.lengthSquared() < 0.001f)
                tangent = QVector3D(1.0f, 0.0f, 0.0f);
            QVector3D axisA = QVector3D::crossProduct(tangent, QVector3D(0.0f, 0.0f, 1.0f)).normalized();
            if (axisA.lengthSquared() < 0.001f)
                axisA = QVector3D(0.0f, 1.0f, 0.0f);
            const QVector3D axisB = QVector3D::crossProduct(tangent, axisA).normalized();
            const float localRadius = radius * (1.0f - t * 0.62f);
            for (int side = 0; side <= sides; ++side) {
                const float angle = float(M_PI * 2.0) * float(side) / float(sides);
                vertices.append(center + axisA * qCos(angle) * localRadius + axisB * qSin(angle) * localRadius);
                colors.append(QColor());
            }
        }
        for (int segment = 0; segment < pathSegments; ++segment) {
            for (int side = 0; side < sides; ++side) {
                const quint32 a = quint32(base + segment * (sides + 1) + side);
                const quint32 b = a + quint32(sides + 1);
                indices << a << b << a + 1;
                indices << a + 1 << b << b + 1;
            }
        }
    };

    appendEllipsoid(QVector3D(0.0f, -0.02f, 0.0f), QVector3D(0.53f, 0.73f, 0.31f), 34, 58, 0.3f);
    appendEllipsoid(QVector3D(0.0f, 0.67f, -0.01f), QVector3D(0.34f, 0.34f, 0.25f), 22, 42, 1.2f);
    appendEllipsoid(QVector3D(-0.13f, 0.91f, 0.0f), QVector3D(0.10f, 0.23f, 0.08f), 14, 24, 2.1f);
    appendEllipsoid(QVector3D(0.13f, 0.91f, 0.0f), QVector3D(0.10f, 0.23f, 0.08f), 14, 24, 2.7f);

    const float starts[] = {0.46f, 0.19f, -0.12f, -0.40f};
    for (int side = -1; side <= 1; side += 2) {
        for (int leg = 0; leg < 4; ++leg) {
            const float direction = float(side);
            const float startY = starts[leg];
            const float endY = startY + (leg < 2 ? 0.30f - leg * 0.08f : -0.22f - (leg - 2) * 0.12f);
            appendCurvedTube(QVector3D(direction * 0.38f, startY, 0.0f),
                             QVector3D(direction * (0.76f + leg * 0.06f), startY + (leg < 2 ? 0.12f : -0.10f), 0.04f),
                             QVector3D(direction * (1.14f + leg * 0.09f), endY, -0.05f),
                             0.062f,
                             13,
                             8);
        }
    }

    normalizeGeometry();
    calculateNormals();
    ensureColors();
}

bool ModelData::savePly(const QString &fileName, QString *error) const
{
    if (vertices.isEmpty()) {
        setError(error, QStringLiteral("没有可保存的模型"));
        return false;
    }

    QSaveFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        setError(error, QStringLiteral("无法创建模型文件：%1").arg(file.errorString()));
        return false;
    }

    QTextStream stream(&file);
    stream.setCodec("UTF-8");
    stream << "ply\n";
    stream << "format ascii 1.0\n";
    stream << "comment Generated by SAM 3D Qt Client\n";
    stream << "element vertex " << vertices.size() << "\n";
    stream << "property float x\nproperty float y\nproperty float z\n";
    stream << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stream << "element face " << triangleCount() << "\n";
    stream << "property list uchar int vertex_indices\n";
    stream << "end_header\n";

    for (int i = 0; i < vertices.size(); ++i) {
        const QVector3D &vertex = vertices.at(i);
        const QColor color = i < colors.size() && colors.at(i).isValid()
                                 ? colors.at(i)
                                 : QColor(16, 178, 119);
        stream << QString::number(vertex.x(), 'f', 7) << ' '
               << QString::number(vertex.y(), 'f', 7) << ' '
               << QString::number(vertex.z(), 'f', 7) << ' '
               << color.red() << ' ' << color.green() << ' ' << color.blue() << '\n';
    }
    for (int i = 0; i + 2 < indices.size(); i += 3)
        stream << "3 " << indices.at(i) << ' ' << indices.at(i + 1) << ' ' << indices.at(i + 2) << '\n';

    if (!file.commit()) {
        setError(error, QStringLiteral("写入模型文件失败：%1").arg(file.errorString()));
        return false;
    }
    return true;
}
