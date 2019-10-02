#include "torchtestdlg.h"

torchTestdlg::torchTestdlg()
{

}

void torchTestdlg::paintEvent(QPaintEvent*)
{
    QPainter painter(this);

    if(m_image.isNull())
        return;
    painter.drawPixmap(this->rect(), QPixmap::fromImage(m_image));
}
