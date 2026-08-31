#include "modeldata.h"

#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMatrix3x3>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QSet>
#include <QVector2D>
#include <QVector4D>
#include <QtEndian>
#include <QtMath>

#include <cstring>
#include <functional>
#include <limits>

namespace {

constexpr quint32 GlbMagic = 0x46546c67;
constexpr quint32 JsonChunk = 0x4e4f534a;
constexpr quint32 BinaryChunk = 0x004e4942;

void setError(QString *error, const QString &message)
{
    if (error)
        *error = message;
}

bool readUnsigned32(const QByteArray &data, int offset, quint32 &value)
{
    if (offset < 0 || offset + 4 > data.size())
        return false;
    quint32 stored = 0;
    std::memcpy(&stored, data.constData() + offset, sizeof(stored));
    value = qFromLittleEndian(stored);
    return true;
}

int componentSize(int componentType)
{
    switch (componentType) {
    case 5120:
    case 5121:
        return 1;
    case 5122:
    case 5123:
        return 2;
    case 5125:
    case 5126:
        return 4;
    default:
        return 0;
    }
}

int componentCount(const QString &type)
{
    if (type == QStringLiteral("SCALAR")) return 1;
    if (type == QStringLiteral("VEC2")) return 2;
    if (type == QStringLiteral("VEC3")) return 3;
    if (type == QStringLiteral("VEC4")) return 4;
    if (type == QStringLiteral("MAT2")) return 4;
    if (type == QStringLiteral("MAT3")) return 9;
    if (type == QStringLiteral("MAT4")) return 16;
    return 0;
}

template<typename T>
T readLittleEndian(const char *source)
{
    T stored{};
    std::memcpy(&stored, source, sizeof(T));
    return qFromLittleEndian(stored);
}

double readComponent(const char *source, int componentType, bool normalized)
{
    switch (componentType) {
    case 5120: {
        const qint8 value = *reinterpret_cast<const qint8 *>(source);
        return normalized ? qMax(-1.0, double(value) / 127.0) : double(value);
    }
    case 5121: {
        const quint8 value = *reinterpret_cast<const quint8 *>(source);
        return normalized ? double(value) / 255.0 : double(value);
    }
    case 5122: {
        const qint16 value = readLittleEndian<qint16>(source);
        return normalized ? qMax(-1.0, double(value) / 32767.0) : double(value);
    }
    case 5123: {
        const quint16 value = readLittleEndian<quint16>(source);
        return normalized ? double(value) / 65535.0 : double(value);
    }
    case 5125: {
        const quint32 value = readLittleEndian<quint32>(source);
        return normalized ? double(value) / 4294967295.0 : double(value);
    }
    case 5126: {
        const quint32 bits = readLittleEndian<quint32>(source);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    default:
        return 0.0;
    }
}

struct AccessorView {
    const QByteArray *buffer = nullptr;
    int offset = 0;
    int stride = 0;
    int count = 0;
    int components = 0;
    int componentType = 0;
    bool normalized = false;
};

bool accessorView(const QJsonObject &root,
                  const QVector<QByteArray> &buffers,
                  int accessorIndex,
                  AccessorView &result,
                  QString *error)
{
    const QJsonArray accessors = root.value(QStringLiteral("accessors")).toArray();
    const QJsonArray views = root.value(QStringLiteral("bufferViews")).toArray();
    if (accessorIndex < 0 || accessorIndex >= accessors.size()) {
        setError(error, QStringLiteral("GLB accessor 索引越界"));
        return false;
    }

    const QJsonObject accessor = accessors.at(accessorIndex).toObject();
    if (accessor.contains(QStringLiteral("sparse"))) {
        setError(error, QStringLiteral("当前 GLB 解析器不支持 sparse accessor"));
        return false;
    }
    const int viewIndex = accessor.value(QStringLiteral("bufferView")).toInt(-1);
    if (viewIndex < 0 || viewIndex >= views.size()) {
        setError(error, QStringLiteral("GLB accessor 缺少有效 bufferView"));
        return false;
    }

    const QJsonObject view = views.at(viewIndex).toObject();
    const int bufferIndex = view.value(QStringLiteral("buffer")).toInt(-1);
    if (bufferIndex < 0 || bufferIndex >= buffers.size()) {
        setError(error, QStringLiteral("GLB bufferView 引用了无效 buffer"));
        return false;
    }

    result.buffer = &buffers.at(bufferIndex);
    result.componentType = accessor.value(QStringLiteral("componentType")).toInt();
    result.components = componentCount(accessor.value(QStringLiteral("type")).toString());
    result.count = accessor.value(QStringLiteral("count")).toInt();
    result.normalized = accessor.value(QStringLiteral("normalized")).toBool(false);
    const int packedStride = componentSize(result.componentType) * result.components;
    result.stride = view.value(QStringLiteral("byteStride")).toInt(packedStride);
    result.offset = view.value(QStringLiteral("byteOffset")).toInt()
                    + accessor.value(QStringLiteral("byteOffset")).toInt();
    const int viewEnd = view.value(QStringLiteral("byteOffset")).toInt()
                        + view.value(QStringLiteral("byteLength")).toInt();

    if (result.count <= 0 || result.components <= 0 || packedStride <= 0
        || result.stride < packedStride || result.offset < 0) {
        setError(error, QStringLiteral("GLB accessor 布局无效"));
        return false;
    }
    const qint64 requiredEnd = qint64(result.offset)
                               + qint64(result.count - 1) * result.stride
                               + packedStride;
    if (requiredEnd > result.buffer->size() || requiredEnd > viewEnd) {
        setError(error, QStringLiteral("GLB accessor 数据越界"));
        return false;
    }
    return true;
}

QVector4D accessorValue(const AccessorView &view, int index)
{
    QVector4D value(0.0f, 0.0f, 0.0f, 1.0f);
    if (!view.buffer || index < 0 || index >= view.count)
        return value;
    const char *source = view.buffer->constData() + view.offset + index * view.stride;
    const int size = componentSize(view.componentType);
    for (int component = 0; component < qMin(4, view.components); ++component)
        value[component] = float(readComponent(source + component * size,
                                               view.componentType,
                                               view.normalized));
    return value;
}

quint32 accessorIndexValue(const AccessorView &view, int index)
{
    if (!view.buffer || index < 0 || index >= view.count)
        return std::numeric_limits<quint32>::max();
    const char *source = view.buffer->constData() + view.offset + index * view.stride;
    switch (view.componentType) {
    case 5121: return *reinterpret_cast<const quint8 *>(source);
    case 5123: return readLittleEndian<quint16>(source);
    case 5125: return readLittleEndian<quint32>(source);
    default: return std::numeric_limits<quint32>::max();
    }
}

QByteArray decodeDataUri(const QString &uri)
{
    const int comma = uri.indexOf(QLatin1Char(','));
    if (!uri.startsWith(QStringLiteral("data:")) || comma < 0)
        return {};
    const QString metadata = uri.left(comma);
    const QByteArray payload = uri.mid(comma + 1).toLatin1();
    return metadata.contains(QStringLiteral(";base64"))
               ? QByteArray::fromBase64(payload)
               : QByteArray::fromPercentEncoding(payload);
}

QVector<QByteArray> loadBuffers(const QJsonObject &root,
                                const QByteArray &binaryChunk,
                                QString *error)
{
    QVector<QByteArray> result;
    const QJsonArray definitions = root.value(QStringLiteral("buffers")).toArray();
    for (int index = 0; index < definitions.size(); ++index) {
        const QJsonObject definition = definitions.at(index).toObject();
        QByteArray bytes;
        const QString uri = definition.value(QStringLiteral("uri")).toString();
        if (uri.isEmpty() && index == 0)
            bytes = binaryChunk;
        else
            bytes = decodeDataUri(uri);
        const int expectedLength = definition.value(QStringLiteral("byteLength")).toInt();
        if (bytes.isEmpty() || expectedLength < 0 || bytes.size() < expectedLength) {
            setError(error, QStringLiteral("GLB buffer %1 数据不完整").arg(index));
            return {};
        }
        result.append(bytes);
    }
    if (result.isEmpty())
        setError(error, QStringLiteral("GLB 没有可用 buffer"));
    return result;
}

QByteArray bufferViewBytes(const QJsonObject &root,
                           const QVector<QByteArray> &buffers,
                           int viewIndex)
{
    const QJsonArray views = root.value(QStringLiteral("bufferViews")).toArray();
    if (viewIndex < 0 || viewIndex >= views.size())
        return {};
    const QJsonObject view = views.at(viewIndex).toObject();
    const int bufferIndex = view.value(QStringLiteral("buffer")).toInt(-1);
    const int offset = view.value(QStringLiteral("byteOffset")).toInt();
    const int length = view.value(QStringLiteral("byteLength")).toInt();
    if (bufferIndex < 0 || bufferIndex >= buffers.size() || offset < 0 || length <= 0
        || offset + length > buffers.at(bufferIndex).size())
        return {};
    return buffers.at(bufferIndex).mid(offset, length);
}

struct TextureInfo {
    QImage image;
    int wrapS = 10497;
    int wrapT = 10497;
};

QVector<TextureInfo> loadTextures(const QJsonObject &root,
                                  const QVector<QByteArray> &buffers)
{
    QVector<QImage> images;
    const QJsonArray imageDefinitions = root.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : imageDefinitions) {
        const QJsonObject definition = value.toObject();
        QByteArray bytes;
        if (definition.contains(QStringLiteral("bufferView")))
            bytes = bufferViewBytes(root, buffers,
                                    definition.value(QStringLiteral("bufferView")).toInt(-1));
        else
            bytes = decodeDataUri(definition.value(QStringLiteral("uri")).toString());
        images.append(QImage::fromData(bytes));
    }

    const QJsonArray samplers = root.value(QStringLiteral("samplers")).toArray();
    const QJsonArray textureDefinitions = root.value(QStringLiteral("textures")).toArray();
    QVector<TextureInfo> textures;
    for (const QJsonValue &value : textureDefinitions) {
        const QJsonObject definition = value.toObject();
        TextureInfo texture;
        const int source = definition.value(QStringLiteral("source")).toInt(-1);
        if (source >= 0 && source < images.size())
            texture.image = images.at(source).convertToFormat(QImage::Format_RGBA8888);
        const int samplerIndex = definition.value(QStringLiteral("sampler")).toInt(-1);
        if (samplerIndex >= 0 && samplerIndex < samplers.size()) {
            const QJsonObject sampler = samplers.at(samplerIndex).toObject();
            texture.wrapS = sampler.value(QStringLiteral("wrapS")).toInt(10497);
            texture.wrapT = sampler.value(QStringLiteral("wrapT")).toInt(10497);
        }
        textures.append(texture);
    }
    return textures;
}

qreal wrapCoordinate(qreal value, int mode)
{
    if (mode == 33071)
        return qBound<qreal>(0.0, value, 1.0);
    if (mode == 33648) {
        const qreal period = value - qFloor(value / 2.0) * 2.0;
        return period <= 1.0 ? period : 2.0 - period;
    }
    return value - qFloor(value);
}

QColor sampledColor(const TextureInfo &texture, const QVector2D &uv)
{
    if (texture.image.isNull())
        return QColor::fromRgbF(1.0, 1.0, 1.0, 1.0);
    const qreal u = wrapCoordinate(uv.x(), texture.wrapS);
    const qreal v = wrapCoordinate(uv.y(), texture.wrapT);
    const int x = qBound(0, qRound(u * (texture.image.width() - 1)), texture.image.width() - 1);
    const int y = qBound(0, qRound((1.0 - v) * (texture.image.height() - 1)), texture.image.height() - 1);
    return texture.image.pixelColor(x, y);
}

struct MaterialInfo {
    QVector4D factor = QVector4D(1.0f, 1.0f, 1.0f, 1.0f);
    int texture = -1;
};

QVector<MaterialInfo> loadMaterials(const QJsonObject &root)
{
    QVector<MaterialInfo> materials;
    const QJsonArray definitions = root.value(QStringLiteral("materials")).toArray();
    for (const QJsonValue &value : definitions) {
        const QJsonObject pbr = value.toObject()
                                    .value(QStringLiteral("pbrMetallicRoughness"))
                                    .toObject();
        MaterialInfo material;
        const QJsonArray factor = pbr.value(QStringLiteral("baseColorFactor")).toArray();
        if (factor.size() == 4) {
            material.factor = QVector4D(float(factor.at(0).toDouble(1.0)),
                                        float(factor.at(1).toDouble(1.0)),
                                        float(factor.at(2).toDouble(1.0)),
                                        float(factor.at(3).toDouble(1.0)));
        }
        material.texture = pbr.value(QStringLiteral("baseColorTexture"))
                               .toObject()
                               .value(QStringLiteral("index"))
                               .toInt(-1);
        materials.append(material);
    }
    return materials;
}

QMatrix4x4 nodeTransform(const QJsonObject &node)
{
    const QJsonArray matrixValues = node.value(QStringLiteral("matrix")).toArray();
    if (matrixValues.size() == 16) {
        QMatrix4x4 matrix;
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row)
                matrix(row, column) = float(matrixValues.at(column * 4 + row).toDouble());
        }
        return matrix;
    }

    QMatrix4x4 matrix;
    const QJsonArray translation = node.value(QStringLiteral("translation")).toArray();
    if (translation.size() == 3) {
        matrix.translate(float(translation.at(0).toDouble()),
                         float(translation.at(1).toDouble()),
                         float(translation.at(2).toDouble()));
    }
    const QJsonArray rotation = node.value(QStringLiteral("rotation")).toArray();
    if (rotation.size() == 4) {
        matrix.rotate(QQuaternion(float(rotation.at(3).toDouble()),
                                  float(rotation.at(0).toDouble()),
                                  float(rotation.at(1).toDouble()),
                                  float(rotation.at(2).toDouble())));
    }
    const QJsonArray scale = node.value(QStringLiteral("scale")).toArray();
    if (scale.size() == 3) {
        matrix.scale(float(scale.at(0).toDouble(1.0)),
                     float(scale.at(1).toDouble(1.0)),
                     float(scale.at(2).toDouble(1.0)));
    }
    return matrix;
}

QColor combineColor(const QVector4D &factor,
                    const QVector4D &vertex,
                    const QColor &texture)
{
    return QColor::fromRgbF(qBound(0.0, double(factor.x() * vertex.x() * texture.redF()), 1.0),
                            qBound(0.0, double(factor.y() * vertex.y() * texture.greenF()), 1.0),
                            qBound(0.0, double(factor.z() * vertex.z() * texture.blueF()), 1.0),
                            1.0);
}

bool appendPrimitive(ModelData &model,
                     const QJsonObject &root,
                     const QVector<QByteArray> &buffers,
                     const QVector<TextureInfo> &textures,
                     const QVector<MaterialInfo> &materials,
                     const QJsonObject &primitive,
                     const QMatrix4x4 &transform,
                     QString *error)
{
    const int mode = primitive.value(QStringLiteral("mode")).toInt(4);
    if (mode != 4 && mode != 5 && mode != 6)
        return true;

    const QJsonObject attributes = primitive.value(QStringLiteral("attributes")).toObject();
    AccessorView positions;
    if (!accessorView(root, buffers,
                      attributes.value(QStringLiteral("POSITION")).toInt(-1),
                      positions, error))
        return false;
    if (positions.components < 3) {
        setError(error, QStringLiteral("GLB POSITION accessor 不是 VEC3"));
        return false;
    }

    AccessorView normalAccessor;
    bool hasNormals = false;
    if (attributes.contains(QStringLiteral("NORMAL"))) {
        hasNormals = accessorView(root, buffers,
                                  attributes.value(QStringLiteral("NORMAL")).toInt(-1),
                                  normalAccessor, error);
        if (!hasNormals)
            return false;
        hasNormals = normalAccessor.count == positions.count && normalAccessor.components >= 3;
    }
    AccessorView colorAccessor;
    bool hasColors = false;
    if (attributes.contains(QStringLiteral("COLOR_0"))) {
        hasColors = accessorView(root, buffers,
                                 attributes.value(QStringLiteral("COLOR_0")).toInt(-1),
                                 colorAccessor, error);
        if (!hasColors)
            return false;
        hasColors = colorAccessor.count == positions.count && colorAccessor.components >= 3;
    }
    AccessorView uvAccessor;
    bool hasUv = false;
    if (attributes.contains(QStringLiteral("TEXCOORD_0"))) {
        hasUv = accessorView(root, buffers,
                             attributes.value(QStringLiteral("TEXCOORD_0")).toInt(-1),
                             uvAccessor, error);
        if (!hasUv)
            return false;
        hasUv = uvAccessor.count == positions.count && uvAccessor.components >= 2;
    }

    const int materialIndex = primitive.value(QStringLiteral("material")).toInt(-1);
    const MaterialInfo material = materialIndex >= 0 && materialIndex < materials.size()
                                      ? materials.at(materialIndex)
                                      : MaterialInfo();
    const TextureInfo texture = material.texture >= 0 && material.texture < textures.size()
                                    ? textures.at(material.texture)
                                    : TextureInfo();

    const int baseVertex = model.vertices.size();
    const QMatrix3x3 normalMatrix = transform.normalMatrix();
    for (int index = 0; index < positions.count; ++index) {
        const QVector4D sourcePosition = accessorValue(positions, index);
        model.vertices.append(transform.map(QVector3D(sourcePosition.x(),
                                                       sourcePosition.y(),
                                                       sourcePosition.z())));

        QVector3D normal;
        if (hasNormals) {
            const QVector4D value = accessorValue(normalAccessor, index);
            normal = QVector3D(normalMatrix(0, 0) * value.x()
                                   + normalMatrix(0, 1) * value.y()
                                   + normalMatrix(0, 2) * value.z(),
                               normalMatrix(1, 0) * value.x()
                                   + normalMatrix(1, 1) * value.y()
                                   + normalMatrix(1, 2) * value.z(),
                               normalMatrix(2, 0) * value.x()
                                   + normalMatrix(2, 1) * value.y()
                                   + normalMatrix(2, 2) * value.z());
            if (!qFuzzyIsNull(normal.lengthSquared()))
                normal.normalize();
        }
        model.normals.append(normal);

        QVector4D vertexColor(1.0f, 1.0f, 1.0f, 1.0f);
        if (hasColors)
            vertexColor = accessorValue(colorAccessor, index);
        QColor textureColor = QColor::fromRgbF(1.0, 1.0, 1.0, 1.0);
        if (hasUv && !texture.image.isNull()) {
            const QVector4D uv = accessorValue(uvAccessor, index);
            textureColor = sampledColor(texture, QVector2D(uv.x(), uv.y()));
        }
        model.colors.append(combineColor(material.factor, vertexColor, textureColor));
    }

    QVector<quint32> sourceIndices;
    if (primitive.contains(QStringLiteral("indices"))) {
        AccessorView indices;
        if (!accessorView(root, buffers,
                          primitive.value(QStringLiteral("indices")).toInt(-1),
                          indices, error))
            return false;
        if (indices.components != 1
            || (indices.componentType != 5121
                && indices.componentType != 5123
                && indices.componentType != 5125)) {
            setError(error, QStringLiteral("GLB indices accessor 类型不受支持"));
            return false;
        }
        sourceIndices.reserve(indices.count);
        for (int index = 0; index < indices.count; ++index)
            sourceIndices.append(accessorIndexValue(indices, index));
    } else {
        sourceIndices.reserve(positions.count);
        for (int index = 0; index < positions.count; ++index)
            sourceIndices.append(quint32(index));
    }

    for (quint32 index : sourceIndices) {
        if (index >= quint32(positions.count)) {
            setError(error, QStringLiteral("GLB primitive 索引越界"));
            return false;
        }
    }

    auto appendTriangle = [&model, baseVertex](quint32 a, quint32 b, quint32 c) {
        if (a == b || b == c || a == c)
            return;
        model.indices << quint32(baseVertex) + a
                      << quint32(baseVertex) + b
                      << quint32(baseVertex) + c;
    };
    if (mode == 4) {
        for (int index = 0; index + 2 < sourceIndices.size(); index += 3)
            appendTriangle(sourceIndices.at(index),
                           sourceIndices.at(index + 1),
                           sourceIndices.at(index + 2));
    } else if (mode == 5) {
        for (int index = 2; index < sourceIndices.size(); ++index) {
            if (index % 2 == 0)
                appendTriangle(sourceIndices.at(index - 2), sourceIndices.at(index - 1), sourceIndices.at(index));
            else
                appendTriangle(sourceIndices.at(index - 1), sourceIndices.at(index - 2), sourceIndices.at(index));
        }
    } else {
        for (int index = 2; index < sourceIndices.size(); ++index)
            appendTriangle(sourceIndices.first(), sourceIndices.at(index - 1), sourceIndices.at(index));
    }
    return true;
}

bool parseGlb(ModelData &model, const QByteArray &data, QString *error)
{
    quint32 magic = 0;
    quint32 version = 0;
    quint32 declaredLength = 0;
    if (!readUnsigned32(data, 0, magic)
        || !readUnsigned32(data, 4, version)
        || !readUnsigned32(data, 8, declaredLength)
        || magic != GlbMagic || version != 2
        || declaredLength < 20 || declaredLength > quint32(data.size())) {
        setError(error, QStringLiteral("文件不是有效的 GLB 2.0 模型"));
        return false;
    }

    QByteArray jsonChunk;
    QByteArray binaryChunk;
    int offset = 12;
    while (offset + 8 <= int(declaredLength)) {
        quint32 length = 0;
        quint32 type = 0;
        if (!readUnsigned32(data, offset, length)
            || !readUnsigned32(data, offset + 4, type)
            || qint64(offset) + 8 + length > declaredLength) {
            setError(error, QStringLiteral("GLB chunk 长度无效"));
            return false;
        }
        const QByteArray chunk = data.mid(offset + 8, int(length));
        if (type == JsonChunk && jsonChunk.isEmpty())
            jsonChunk = chunk;
        else if (type == BinaryChunk && binaryChunk.isEmpty())
            binaryChunk = chunk;
        offset += 8 + int(length);
    }
    if (jsonChunk.isEmpty()) {
        setError(error, QStringLiteral("GLB 缺少 JSON chunk"));
        return false;
    }

    while (!jsonChunk.isEmpty()
           && (jsonChunk.endsWith(' ') || jsonChunk.endsWith('\0')))
        jsonChunk.chop(1);
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(jsonChunk, &parseError);
    if (!document.isObject()) {
        setError(error, QStringLiteral("GLB JSON 无法解析：%1").arg(parseError.errorString()));
        return false;
    }

    const QJsonObject root = document.object();
    const QVector<QByteArray> buffers = loadBuffers(root, binaryChunk, error);
    if (buffers.isEmpty())
        return false;
    const QVector<TextureInfo> textures = loadTextures(root, buffers);
    const QVector<MaterialInfo> materials = loadMaterials(root);
    const QJsonArray meshes = root.value(QStringLiteral("meshes")).toArray();
    const QJsonArray nodes = root.value(QStringLiteral("nodes")).toArray();

    auto appendMesh = [&](int meshIndex, const QMatrix4x4 &transform) -> bool {
        if (meshIndex < 0 || meshIndex >= meshes.size()) {
            setError(error, QStringLiteral("GLB node 引用了无效 mesh"));
            return false;
        }
        const QJsonArray primitives = meshes.at(meshIndex)
                                          .toObject()
                                          .value(QStringLiteral("primitives"))
                                          .toArray();
        for (const QJsonValue &primitive : primitives) {
            if (!appendPrimitive(model, root, buffers, textures, materials,
                                 primitive.toObject(), transform, error))
                return false;
        }
        return true;
    };

    bool traversalOk = true;
    QSet<int> activeNodes;
    std::function<void(int, const QMatrix4x4 &)> visitNode;
    visitNode = [&](int nodeIndex, const QMatrix4x4 &parentTransform) {
        if (!traversalOk)
            return;
        if (nodeIndex < 0 || nodeIndex >= nodes.size() || activeNodes.contains(nodeIndex)) {
            setError(error, QStringLiteral("GLB 场景节点结构无效"));
            traversalOk = false;
            return;
        }
        activeNodes.insert(nodeIndex);
        const QJsonObject node = nodes.at(nodeIndex).toObject();
        const QMatrix4x4 transform = parentTransform * nodeTransform(node);
        if (node.contains(QStringLiteral("mesh"))
            && !appendMesh(node.value(QStringLiteral("mesh")).toInt(-1), transform)) {
            traversalOk = false;
        }
        const QJsonArray children = node.value(QStringLiteral("children")).toArray();
        for (const QJsonValue &child : children)
            visitNode(child.toInt(-1), transform);
        activeNodes.remove(nodeIndex);
    };

    if (!nodes.isEmpty()) {
        QJsonArray roots;
        const QJsonArray scenes = root.value(QStringLiteral("scenes")).toArray();
        const int sceneIndex = root.value(QStringLiteral("scene")).toInt(0);
        if (sceneIndex >= 0 && sceneIndex < scenes.size())
            roots = scenes.at(sceneIndex).toObject().value(QStringLiteral("nodes")).toArray();
        if (roots.isEmpty()) {
            QSet<int> children;
            for (const QJsonValue &nodeValue : nodes) {
                for (const QJsonValue &child : nodeValue.toObject().value(QStringLiteral("children")).toArray())
                    children.insert(child.toInt(-1));
            }
            for (int index = 0; index < nodes.size(); ++index) {
                if (!children.contains(index))
                    roots.append(index);
            }
        }
        for (const QJsonValue &rootNode : roots)
            visitNode(rootNode.toInt(-1), QMatrix4x4());
    } else {
        for (int meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
            if (!appendMesh(meshIndex, QMatrix4x4()))
                return false;
        }
    }

    if (!traversalOk)
        return false;
    if (model.vertices.isEmpty() || model.indices.isEmpty()) {
        setError(error, QStringLiteral("GLB 中没有可渲染的三角网格"));
        return false;
    }
    return true;
}

} // namespace

void ModelData::clear()
{
    vertices.clear();
    normals.clear();
    colors.clear();
    indices.clear();
    m_glbData.clear();
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
    if (QFileInfo(fileName).suffix().compare(QStringLiteral("glb"), Qt::CaseInsensitive) != 0) {
        setError(error, QStringLiteral("仅支持 GLB 2.0 模型"));
        return false;
    }
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        setError(error, QStringLiteral("无法打开模型文件：%1").arg(file.errorString()));
        return false;
    }
    return loadGlbData(file.readAll(), error);
}

bool ModelData::loadGlbData(const QByteArray &data, QString *error)
{
    clear();
    if (!parseGlb(*this, data, error)) {
        clear();
        return false;
    }
    normalizeGeometry();

    bool needsNormals = normals.size() != vertices.size();
    if (!needsNormals) {
        for (const QVector3D &normal : normals) {
            if (qFuzzyIsNull(normal.lengthSquared())) {
                needsNormals = true;
                break;
            }
        }
    }
    if (needsNormals)
        calculateNormals();
    ensureColors();
    m_glbData = data;
    return true;
}

bool ModelData::saveGlb(const QString &fileName, QString *error) const
{
    if (m_glbData.isEmpty()) {
        setError(error, QStringLiteral("当前模型没有可导出的原始 GLB 数据"));
        return false;
    }
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly)) {
        setError(error, QStringLiteral("无法创建 GLB 文件：%1").arg(file.errorString()));
        return false;
    }
    if (file.write(m_glbData) != m_glbData.size()) {
        setError(error, QStringLiteral("GLB 文件写入不完整"));
        return false;
    }
    return true;
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
    for (const QVector3D &vertex : vertices) {
        minimum.setX(qMin(minimum.x(), vertex.x()));
        minimum.setY(qMin(minimum.y(), vertex.y()));
        minimum.setZ(qMin(minimum.z(), vertex.z()));
        maximum.setX(qMax(maximum.x(), vertex.x()));
        maximum.setY(qMax(maximum.y(), vertex.y()));
        maximum.setZ(qMax(maximum.z(), vertex.z()));
    }
    const QVector3D center = (minimum + maximum) * 0.5f;
    const QVector3D extent = maximum - minimum;
    const float largest = qMax(extent.x(), qMax(extent.y(), extent.z()));
    const float scale = largest > 0.000001f ? 1.8f / largest : 1.0f;
    for (QVector3D &vertex : vertices)
        vertex = (vertex - center) * scale;
}

void ModelData::calculateNormals()
{
    normals.fill(QVector3D(), vertices.size());
    for (int index = 0; index + 2 < indices.size(); index += 3) {
        const quint32 ia = indices.at(index);
        const quint32 ib = indices.at(index + 1);
        const quint32 ic = indices.at(index + 2);
        if (ia >= quint32(vertices.size())
            || ib >= quint32(vertices.size())
            || ic >= quint32(vertices.size()))
            continue;
        const QVector3D normal = QVector3D::crossProduct(vertices.at(int(ib)) - vertices.at(int(ia)),
                                                         vertices.at(int(ic)) - vertices.at(int(ia)));
        normals[int(ia)] += normal;
        normals[int(ib)] += normal;
        normals[int(ic)] += normal;
    }
    for (int index = 0; index < normals.size(); ++index) {
        if (qFuzzyIsNull(normals[index].lengthSquared()))
            normals[index] = vertices.at(index).normalized();
        else
            normals[index].normalize();
    }
}

void ModelData::ensureColors()
{
    while (colors.size() < vertices.size())
        colors.append(QColor(152, 184, 112));
    if (colors.size() > vertices.size())
        colors.resize(vertices.size());
}

void ModelData::createOrganicSample()
{
    clear();
    constexpr int rings = 36;
    constexpr int sectors = 64;
    for (int ring = 0; ring <= rings; ++ring) {
        const float v = float(ring) / rings;
        const float latitude = float(M_PI) * (v - 0.5f);
        for (int sector = 0; sector <= sectors; ++sector) {
            const float u = float(sector) / sectors;
            const float longitude = float(M_PI * 2.0) * u;
            const float texture = 1.0f + 0.055f * qSin(longitude * 7.0f + latitude * 5.0f);
            QVector3D point(qCos(latitude) * qCos(longitude) * 0.72f,
                            qSin(latitude) * 0.58f,
                            qCos(latitude) * qSin(longitude) * 0.86f);
            point *= texture;
            vertices.append(point);
            const float shade = 0.78f + 0.18f * qSin(longitude * 3.0f + latitude);
            colors.append(QColor::fromRgbF(0.37f * shade, 0.56f * shade, 0.22f * shade));
        }
    }
    for (int ring = 0; ring < rings; ++ring) {
        for (int sector = 0; sector < sectors; ++sector) {
            const quint32 a = quint32(ring * (sectors + 1) + sector);
            const quint32 b = a + quint32(sectors + 1);
            indices << a << b << a + 1;
            indices << a + 1 << b << b + 1;
        }
    }
    calculateNormals();
}
