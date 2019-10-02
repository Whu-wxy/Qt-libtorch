#ifndef TORCHTESTDLG_H
#define TORCHTESTDLG_H

#include <QObject>
#include <QDialog>
#include <QImage>
#include <QPainter>

class torchTestdlg : public QDialog
{
public:
    torchTestdlg();

    void    setImage(QImage img) { m_image = img;
                                   update(); }
    void    paintEvent(QPaintEvent*);

    QImage m_image;

};

#endif // TORCHTESTDLG_H
