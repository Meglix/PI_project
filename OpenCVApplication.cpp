#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <string>

wchar_t* projectPath;

struct TextBlock {
    Rect box;
    Mat mask;
    std::string text;
    float confidence;
};

class AdvancedBookProcessor {
private:
    Mat original;
    Mat processed;
    Mat textMask;
    Mat background;
    Mat final;
    std::vector<TextBlock> blocks;

public:
    bool loadImage(const std::string& path) {
        original = imread(path, IMREAD_COLOR);
        if (original.empty()) {
            printf("Cannot load image: %s\n", path.c_str());
            return false;
        }
        processed = original.clone();
        return true;
    }

    // convertim grayscale, detectam text
    void findTextBlocks() {
        printf("Advanced text detection...\n");

        Mat gray;
        cvtColor(processed, gray, COLOR_BGR2GRAY);

        // Method 1: Edge-based detection
        Mat edgesMask = detectTextByEdges(gray);

        // Method 2: Gradient-based detection  
        Mat gradientMask = detectTextByGradient(gray);

        // Method 3: Variance-based detection
        Mat varianceMask = detectTextByVariance(gray);

        Mat combinedMask = Mat::zeros(gray.size(), CV_8UC1);
        for (int i = 0; i < gray.rows; i++) {
            for (int j = 0; j < gray.cols; j++) {
                int score = 0;
                if (edgesMask.at<uchar>(i, j) > 0) score++;
                if (gradientMask.at<uchar>(i, j) > 0) score++;
                if (varianceMask.at<uchar>(i, j) > 0) score++;

                if (score >= 2) {
                    combinedMask.at<uchar>(i, j) = 255;
                }
            }
        }
        //daca trece peste 2 metode e text
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(combinedMask, combinedMask, MORPH_CLOSE, kernel);
        morphologyEx(combinedMask, combinedMask, MORPH_OPEN, kernel);

        extractTextBlocks(combinedMask);

        printf("Found %d text blocks\n", (int)blocks.size());
    }
    //alg canny
    Mat detectTextByEdges(const Mat& gray) {
        Mat edges;
        Canny(gray, edges, 50, 150);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(edges, edges, kernel, Point(-1, -1), 2);

        return edges;
    }

    //alg sobel
    Mat detectTextByGradient(const Mat& gray) {
        Mat gradX = Mat::zeros(gray.size(), CV_32F);
        Mat gradY = Mat::zeros(gray.size(), CV_32F);

        int sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        int sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

        for (int i = 1; i < gray.rows - 1; i++) {
            for (int j = 1; j < gray.cols - 1; j++) {
                float gx = 0, gy = 0;
                for (int u = 0; u < 3; u++) {
                    for (int v = 0; v < 3; v++) {
                        uchar pixel = gray.at<uchar>(i + u - 1, j + v - 1);
                        gx += pixel * sobelX[u][v];
                        gy += pixel * sobelY[u][v];
                    }
                }
                gradX.at<float>(i, j) = gx;
                gradY.at<float>(i, j) = gy;
            }
        }

        // Calc magnitude
        Mat magnitude = Mat::zeros(gray.size(), CV_8UC1);
        for (int i = 0; i < gray.rows; i++) {
            for (int j = 0; j < gray.cols; j++) {
                float gx = gradX.at<float>(i, j);
                float gy = gradY.at<float>(i, j);
                float mag = sqrt(gx * gx + gy * gy);
                magnitude.at<uchar>(i, j) = saturate_cast<uchar>(mag / 4.0);
            }
        }

        Mat result;
        threshold(magnitude, result, 30, 255, THRESH_BINARY);
        return result;
    }
    //zonele cu o variate mai mare =>text !!!!!!!!!!!!!buguribuguribuguri
    Mat detectTextByVariance(const Mat& gray) {
        Mat result = Mat::zeros(gray.size(), CV_8UC1);
        int windowSize = 15;
        int k = windowSize / 2;

        for (int i = k; i < gray.rows - k; i++) {
            for (int j = k; j < gray.cols - k; j++) {
                float mean = 0;
                for (int u = -k; u <= k; u++) {
                    for (int v = -k; v <= k; v++) {
                        mean += gray.at<uchar>(i + u, j + v);
                    }
                }
                mean /= (windowSize * windowSize);

                float variance = 0;
                for (int u = -k; u <= k; u++) {
                    for (int v = -k; v <= k; v++) {
                        float diff = gray.at<uchar>(i + u, j + v) - mean;
                        variance += diff * diff;
                    }
                }
                variance /= (windowSize * windowSize);

                if (variance > 800) {
                    result.at<uchar>(i, j) = 255;
                }
            }
        }

        return result;
    }
    //prelucreaza blocurile de text
    void extractTextBlocks(const Mat& mask) {
        blocks.clear();

        std::vector<std::vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (auto& contour : contours) {
            Rect box = boundingRect(contour);

            float aspectRatio = (float)box.width / box.height;
            if (box.width > 15 && box.height > 8 &&
                box.width < 600 && box.height < 150 &&
                box.area() > 120 &&
                aspectRatio > 0.3 && aspectRatio < 20) {

                TextBlock block;
                block.box = box;
                block.mask = Mat::zeros(original.size(), CV_8UC1);
                drawContours(block.mask, std::vector<std::vector<Point>>{contour}, -1, Scalar(255), -1);
                block.confidence = calculateTextConfidence(box, mask);

                if (block.confidence > 0.3) {
                    block.text = "Sample text " + std::to_string(blocks.size() + 1);
                    blocks.push_back(block);
                }
            }
        }

        groupNearbyBlocks();
    }

    float calculateTextConfidence(const Rect& box, const Mat& mask) {
        int textPixels = 0;
        int totalPixels = box.width * box.height;

        for (int i = box.y; i < box.y + box.height; i++) {
            for (int j = box.x; j < box.x + box.width; j++) {
                if (mask.at<uchar>(i, j) > 0) {
                    textPixels++;
                }
            }
        }
        //returnam 0-1 cat de sigur este text
        float density = (float)textPixels / totalPixels;
        float densityScore = 1.0f - abs(density - 0.3f) / 0.3f;
        densityScore = max(0.0f, densityScore);

        float aspectRatio = (float)box.width / box.height;
        float aspectScore = 1.0f;
        if (aspectRatio < 1.0f) aspectScore = aspectRatio;
        if (aspectRatio > 10.0f) aspectScore = 10.0f / aspectRatio;

        float sizeScore = min(1.0f, (float)box.area() / 1000.0f);
        //formula vietii
        return (densityScore + aspectScore + sizeScore) / 3.0f;
    }

    void groupNearbyBlocks() {
        if (blocks.size() < 2) return;

        std::vector<TextBlock> grouped;
        std::vector<bool> used(blocks.size(), false);

        for (size_t i = 0; i < blocks.size(); i++) {
            if (used[i]) continue;

            std::vector<int> group;
            group.push_back(i);
            used[i] = true;

            for (size_t j = i + 1; j < blocks.size(); j++) {
                if (used[j]) continue;

                Rect r1 = blocks[i].box;
                Rect r2 = blocks[j].box;

                int verticalOverlap = min(r1.y + r1.height, r2.y + r2.height) - max(r1.y, r2.y);
                int minHeight = min(r1.height, r2.height);

                if (verticalOverlap > minHeight * 0.5) {
                    int horizontalGap = min(abs(r1.x + r1.width - r2.x), abs(r2.x + r2.width - r1.x));
                    if (horizontalGap < minHeight * 1.5) {
                        group.push_back(j);
                        used[j] = true;
                    }
                }
            }
            //cuvintele dintr o propozitie au acelasi bloc
            if (group.size() == 1) {
                grouped.push_back(blocks[i]);
            }
            else {
                TextBlock merged;
                merged.box = blocks[group[0]].box;
                merged.confidence = 0;

                for (int idx : group) {
                    merged.box = merged.box | blocks[idx].box;
                    merged.confidence += blocks[idx].confidence;
                }
                merged.confidence /= group.size();

                merged.mask = Mat::zeros(original.size(), CV_8UC1);
                for (int idx : group) {
                    merged.mask |= blocks[idx].mask;
                }

                merged.text = "Merged text block " + std::to_string(grouped.size() + 1);
                grouped.push_back(merged);
            }
        }

        blocks = grouped;
    }

    // text mask +combinatie de masti
    void createTextMask() {
        printf("Creating precise text mask...\n");

        textMask = Mat::zeros(original.size(), CV_8UC1);
        for (auto& block : blocks) {
            textMask |= block.mask;
        }

        refineTextMask();
    }

    void refineTextMask() {
        Mat gray;
        cvtColor(original, gray, COLOR_BGR2GRAY);
        Mat refined = Mat::zeros(original.size(), CV_8UC1);

        for (int i = 0; i < original.rows; i++) {
            for (int j = 0; j < original.cols; j++) {
                if (textMask.at<uchar>(i, j) > 0) {
                    uchar intensity = gray.at<uchar>(i, j);

                    int count = 0;
                    float localMean = 0;
                    for (int di = -5; di <= 5; di++) {
                        for (int dj = -5; dj <= 5; dj++) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni >= 0 && ni < gray.rows && nj >= 0 && nj < gray.cols) {
                                localMean += gray.at<uchar>(ni, nj);
                                count++;
                            }
                        }
                    }
                    localMean /= count;
                    //pastram doar pixelii cu gradul suficient in cazul asta 25, am incercat cu alte variante
                    //asta e cea mai buna
                    if (abs(intensity - localMean) > 25) {
                        refined.at<uchar>(i, j) = 255;
                    }
                }
            }
        }

        textMask = refined;
    }

    void removeText() { //re move move
        printf("Removing text with inpainting...\n");

        background = original.clone();

        Mat maskDilated;
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        dilate(textMask, maskDilated, kernel, Point(-1, -1), 1);

        inpaint(background, maskDilated, background, 3, INPAINT_TELEA);
    }

    void renderText() {
        printf("Rendering high-quality text...\n");

        final = background.clone();

        for (auto& block : blocks) {
            renderTextBlock(block);
        }
    }

    void renderTextBlock(const TextBlock& block) {
        double fontScale = block.box.height / 35.0;
        fontScale = max(0.4, min(fontScale, 2.0));
        int thickness = max(1, (int)(fontScale * 1.5));

        std::vector<std::string> lines = wrapText(block.text, block.box.width - 10, fontScale, thickness);

        int lineHeight = (int)(fontScale * 25 + 5);
        Point pos(block.box.x + 5, block.box.y + lineHeight);

        for (auto& line : lines) {
            putText(final, line, pos, FONT_HERSHEY_SIMPLEX,
                fontScale, Scalar(0, 0, 0), thickness, LINE_AA);
            pos.y += lineHeight;

            if (pos.y > block.box.y + block.box.height - 5) break;
        }
    }

    std::vector<std::string> wrapText(const std::string& text, int maxWidth, double fontScale, int thickness) {
        std::vector<std::string> lines;

        Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, fontScale, thickness, nullptr);
        if (textSize.width <= maxWidth) {
            lines.push_back(text);
            return lines;
        }

        std::string remaining = text;
        std::string currentLine = "";

        while (!remaining.empty()) {
            size_t spacePos = remaining.find(' ');
            std::string word = (spacePos == std::string::npos) ? remaining : remaining.substr(0, spacePos);

            std::string testLine = currentLine.empty() ? word : currentLine + " " + word;
            Size size = getTextSize(testLine, FONT_HERSHEY_SIMPLEX, fontScale, thickness, nullptr);

            if (size.width <= maxWidth) {
                currentLine = testLine;
                if (spacePos == std::string::npos) {
                    lines.push_back(currentLine);
                    break;
                }
                remaining = remaining.substr(spacePos + 1);
            }
            else {
                if (!currentLine.empty()) {
                    lines.push_back(currentLine);
                    currentLine = "";
                }
                else {
                    lines.push_back(word);
                    if (spacePos == std::string::npos) break;
                    remaining = remaining.substr(spacePos + 1);
                }
            }
        }

        if (!currentLine.empty()) {
            lines.push_back(currentLine);
        }

        return lines;
    }

    void processAll() {
        printf("=== Starting text-background separation ===\n");
        findTextBlocks();
        createTextMask();         ///////////procesam tot
        removeText();
        renderText();
        printf("=== Processing complete ===\n");
    }

    //results
    Mat getOriginal() { return original; }
    Mat getTextMask() { return textMask; }
    Mat getBackground() { return background; }
    Mat getFinal() { return final; }

    Mat showTextBlocks() {
        Mat result = original.clone();
        for (size_t i = 0; i < blocks.size(); i++) {
            Scalar color(0, 255, 0);
            rectangle(result, blocks[i].box, color, 2);

            char label[100];
            sprintf(label, "%d (%.2f)", (int)i + 1, blocks[i].confidence);
            putText(result, label, Point(blocks[i].box.x, blocks[i].box.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        return result;
    }
    // 4x4 imag
    Mat showDetectionSteps() {
        Mat gray;
        cvtColor(original, gray, COLOR_BGR2GRAY);

        Mat edges = detectTextByEdges(gray);
        Mat gradient = detectTextByGradient(gray);
        Mat variance = detectTextByVariance(gray);

        Mat combined = Mat::zeros(gray.rows, gray.cols * 4, CV_8UC3);

        Mat grayCol;
        cvtColor(gray, grayCol, COLOR_GRAY2BGR);
        grayCol.copyTo(combined(Rect(0, 0, gray.cols, gray.rows)));

        Mat edgesCol;
        cvtColor(edges, edgesCol, COLOR_GRAY2BGR);
        edgesCol.copyTo(combined(Rect(gray.cols, 0, gray.cols, gray.rows)));

        Mat gradCol;
        cvtColor(gradient, gradCol, COLOR_GRAY2BGR);
        gradCol.copyTo(combined(Rect(gray.cols * 2, 0, gray.cols, gray.rows)));

        Mat varCol;
        cvtColor(variance, varCol, COLOR_GRAY2BGR);
        varCol.copyTo(combined(Rect(gray.cols * 3, 0, gray.cols, gray.rows)));

        return combined;
    }

    void saveResults(const std::string& baseName) {
        imwrite(baseName + "_1_original.jpg", original);
        imwrite(baseName + "_2_detection_steps.jpg", showDetectionSteps());
        imwrite(baseName + "_3_text_blocks.jpg", showTextBlocks());
        imwrite(baseName + "_4_text_mask.jpg", textMask);
        imwrite(baseName + "_5_background.jpg", background);
        imwrite(baseName + "_6_final.jpg", final);
        printf("Results saved with prefix: %s\n", baseName.c_str());
    }
};

AdvancedBookProcessor processor;

void processSingleImage() {
    char fname[MAX_PATH];
    if (!openFileDlg(fname)) return;

    if (!processor.loadImage(fname)) return;

    processor.processAll();

    Mat orig, steps, blocks, mask, bg, final;
    resizeImg(processor.getOriginal(), orig, 400, true);
    resizeImg(processor.showDetectionSteps(), steps, 800, true);
    resizeImg(processor.showTextBlocks(), blocks, 400, true);
    resizeImg(processor.getTextMask(), mask, 400, true);
    resizeImg(processor.getBackground(), bg, 400, true);
    resizeImg(processor.getFinal(), final, 400, true);

    imshow("1-Original", orig);
    imshow("2-Detection Methods", steps);
    imshow("3-Text Blocks", blocks);
    imshow("4-Text Mask", mask);
    imshow("5-Background", bg);
    imshow("6-Final Result", final);

    waitKey(0);
    destroyAllWindows();

    printf("Save results? (y/n): ");
    char choice;
    scanf(" %c", &choice);
    if (choice == 'y') {
        std::string base = fname;
        size_t dot = base.find_last_of('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        processor.saveResults(base + "_processed");
    }
}

void stepByStep() {
    char fname[MAX_PATH];
    if (!openFileDlg(fname)) return;

    if (!processor.loadImage(fname)) return;

    int step;
    do {
        printf("1 - Show detection methods\n");
        printf("2 - Find text blocks\n");
        printf("3 - Create text mask\n");
        printf("4 - Remove text from background\n");
        printf("5 - Render new text\n");
        printf("6 - Show all results\n");
        printf("0 - Back\n");
        printf("Step: ");
        scanf("%d", &step);

        Mat result;

        switch (step) {
        case 1:
            result = processor.showDetectionSteps();
            break;
        case 2:
            processor.findTextBlocks();
            result = processor.showTextBlocks();
            break;
        case 3:
            processor.createTextMask();
            result = processor.getTextMask();
            break;
        case 4:
            processor.removeText();
            result = processor.getBackground();
            break;
        case 5:
            processor.renderText();
            result = processor.getFinal();
            break;
        case 6:
        {
            Mat orig, blocks, mask, bg, final;
            resizeImg(processor.getOriginal(), orig, 300, true);
            resizeImg(processor.showTextBlocks(), blocks, 300, true);
            resizeImg(processor.getTextMask(), mask, 300, true);
            resizeImg(processor.getBackground(), bg, 300, true);
            resizeImg(processor.getFinal(), final, 300, true);

            imshow("Original", orig);
            imshow("Blocks", blocks);
            imshow("Mask", mask);
            imshow("Background", bg);
            imshow("Final", final);
            waitKey(0);
            destroyAllWindows();
        }
        continue;
        }

        if (step >= 1 && step <= 5 && !result.empty()) {
            Mat resized;
            if (step == 1) {
                resizeImg(result, resized, 800, true);
            }
            else {
                resizeImg(result, resized, 500, true);
            }
            imshow("Result", resized);
            waitKey(0);
            destroyWindow("Result");
        }

    } while (step != 0);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    int op;
    do {
        system("cls");
        destroyAllWindows();

        printf("===== ADVANCED TEXT DETECTION PROCESSOR =====\n");
        printf("Smart text-background separation system\n\n");

        printf("DETECTION METHODS:\n");
        printf("- Canny edge detection for text boundaries\n");
        printf("- Sobel gradient analysis for text patterns\n");
        printf("- Local variance analysis for texture\n");
        printf("- Morphological operations for cleanup\n");
        printf("- Smart grouping of nearby text blocks\n\n"); 

        printf("OPTIONS:\n");
        printf("1 - Process single image\n");
        printf("2 - Step-by-step demo\n");
        printf("0 - Exit\n\n");

        printf("Choose: ");
        scanf("%d", &op);

        switch (op) {
        case 1: processSingleImage(); break;
        case 2: stepByStep(); break;
        }

    } while (op != 0);

    return 0;
}