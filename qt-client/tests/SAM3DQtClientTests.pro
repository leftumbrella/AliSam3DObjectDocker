QT += core gui widgets opengl testlib

TEMPLATE = app
TARGET = SAM3DQtClientTests
CONFIG += console c++17 testcase
CONFIG -= app_bundle

INCLUDEPATH += ../src

SOURCES += \
    test_modeldata.cpp \
    ../src/editorcanvas.cpp \
    ../src/modeldata.cpp

HEADERS += \
    ../src/editorcanvas.h \
    ../src/modeldata.h

RESOURCES += ../resources.qrc

win32:LIBS += opengl32.lib

msvc {
    QMAKE_CFLAGS += /utf-8
    QMAKE_CXXFLAGS += /utf-8
}
