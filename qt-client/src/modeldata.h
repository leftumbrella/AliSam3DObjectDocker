#pragma once

#include <QByteArray>
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
    bool loadGlbData(const QByteArray &data, QString *error = nullptr);
    bool saveGlb(const QString &fileName, QString *error = nullptr) const;
    void createOrganicSample();
    void clear();

    bool isEmpty() const;
    int triangleCount() const;

private:
    void normalizeGeometry();
    void calculateNormals();
    void ensureColors();

    QByteArray m_glbData;
};
