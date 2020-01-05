#include <QApplication>
#include <QImage>
#undef slots
#include "torch/torch.h"
#include "torch/jit.h"
#include "torch/nn.h"
#include "torch/script.h"
#define slots Q_SLOTS

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <time.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

using namespace torch;
using namespace std;
using namespace cv;

bool LoadImage(std::string file_name, cv::Mat &image);

bool LoadImageNetLabel(std::string file_name,
                       std::vector<std::string> &labels);

Mat imgClassifier(QString modelPath, QString labPath, QString img_path);

cv::Mat QImageToMat(QImage image);

QImage MatToQImage(cv::Mat mtx);

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load("/home/wxy/QtWork/QtTorch/model.pt");//model.pt//resnet50.pt

    // assert(module != nullptr);
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({ 1, 3, 224, 224 }));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    Mat result = imgClassifier("/home/wxy/QtWork/QtTorch/resnet50.pt", "/home/wxy/QtWork/QtTorch/label.txt",
                        "/home/wxy/QtWork/QtTorch/pic/shark.jpg");

    imshow("result", result);
    return a.exec();
}



bool LoadImage(std::string file_name, cv::Mat &image)
{
    image = cv::imread(file_name);  // CV_8UC3
    if (image.empty() || !image.data)
        return false;

    cv::cvtColor(image, image, CV_BGR2RGB);
    std::cout << "== image size: " << image.size() << " ==" << std::endl;

    // scale image to fit
    cv::Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
    cv::resize(image, image, scale);
    std::cout << "== simply resize: " << image.size() << " ==" << std::endl;

    // convert [unsigned int] to [float]
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    return true;
}

bool LoadImageNetLabel(std::string file_name,
                       std::vector<std::string> &labels)
{
    std::ifstream ifs(file_name);
    if (!ifs) {
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        labels.push_back(line);
    }
    return true;
}

Mat imgClassifier(QString modelPath, QString labPath, QString img_path)
{
    clock_t start,finish;
    double totaltime;
    start=clock();

    cv::Mat mat2Draw = cv::imread(img_path.toStdString());  // CV_8UC3
    if (mat2Draw.empty() || !mat2Draw.data)
        return mat2Draw;

    cv::Mat image;

    torch::jit::script::Module module = torch::jit::load(modelPath.toStdString());

    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"运行时间:"<<totaltime<<"秒"<<endl;

    // to GPU
    // module.to(at::kCUDA);

    // assert(module != nullptr);
    std::cout << "== ResNet50 loaded!\n";
    std::vector<std::string> labels;
    if (LoadImageNetLabel(labPath.toStdString(), labels))
    {
        std::cout << "== Label loaded! Let's try it\n";
    }
    else
    {
        std::cerr << "Please check your label file path." << std::endl;
        return image;
    }


    if (LoadImage(img_path.toStdString(), image))
    {
        auto input_tensor = torch::from_blob(
                    image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
        input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
        input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

        // to GPU
        //  input_tensor = input_tensor.to(at::kCUDA);

        start=clock();
        torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
        finish=clock();
        totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
        cout<<"运行时间:"<<totaltime<<"秒"<<endl;

        auto results = out_tensor.sort(-1, true);
        auto softmaxs = std::get<0>(results)[0].softmax(0);
        auto indexs = std::get<1>(results)[0];

        std::string topPred = "";
        for (int i = 0; i < kTOP_K; ++i)
        {
            auto idx = indexs[i].item<int>();
            if(i==0)
                topPred = labels[idx];
            std::cout << "    ============= Top-" << i + 1
                      << " =============" << std::endl;
            std::cout << "    Label:  " << labels[idx] << std::endl;
            std::cout << "    With Probability:  "
                      << softmaxs[i].item<float>() * 100.0f << "%" << std::endl;
        }

        if(topPred != "")
            cv::putText(mat2Draw, topPred, cv::Point(50, 50), 1, 2, cv::Scalar(0, 255, 0), 2);

    }
    else
    {
        std::cout << "Can't load the image, please check your path." << std::endl;
    }

    return mat2Draw;
}



cv::Mat QImageToMat(QImage image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Grayscale8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}

QImage MatToQImage(cv::Mat InputMat)
{
    cv::Mat TmpMat;

    // convert the color space to RGB
    if (InputMat.channels() == 1)

    {
        cv::cvtColor(InputMat, TmpMat, CV_GRAY2RGB);
    }

    else

    {
        cv::cvtColor(InputMat, TmpMat, CV_BGR2RGB);
    }


    // construct the QImage using the data of the mat, while do not copy the data

    QImage Result = QImage((const uchar*)(TmpMat.data), TmpMat.cols, TmpMat.rows,

                           QImage::Format_RGB888);

    // deep copy the data from mat to QImage

    Result.bits();

    return Result;

}
