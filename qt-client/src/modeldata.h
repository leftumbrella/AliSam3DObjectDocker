#pragma once

#include <QColor>
#include <QString>
#include <QVector>
#include <QVector3D>

class ModelData
{
public:
    QVector<QVector3D> vertices;
    QVector<QVector3D> normals;
    QVector<QColor> colors;
    QVector<quint32> indices;

    bool load(const QString &fileName, QString *error = nullptr);
    bool savePly(const QString &fileName, QString *error = nullptr) const;
    void createOrganicSample();
    void clear();

    bool isEmpty() const;
    int triangleCount() const;

private:
    bool loadObj(const QByteArray &data, QString *error);
    bool loadPly(const QByteArray &data, QString *error);
    void normalizeGeometry();
    void calculateNormals();
    void ensureColors();
};
