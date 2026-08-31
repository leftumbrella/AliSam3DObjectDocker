QT += core gui widgets network opengl

TEMPLATE = app
TARGET = SAM3DQtClient
CONFIG += c++17
CONFIG -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS
INCLUDEPATH += $$PWD/src

SOURCES += \
    src/main.cpp \
    src/editorcanvas.cpp \
    src/modeldata.cpp \
    src/sam3dclient.cpp

HEADERS += \
    src/editorcanvas.h \
    src/modeldata.h \
    src/sam3dclient.h

RESOURCES += resources.qrc

win32:LIBS += opengl32.lib

msvc {
    QMAKE_CFLAGS += /utf-8
    QMAKE_CXXFLAGS += /utf-8
}
