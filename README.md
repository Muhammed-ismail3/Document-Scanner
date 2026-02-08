# Document Scanner (OpenCV)

A real-time computer vision application that detects documents via webcam, performs perspective correction, and applies image enhancement to simulate a high-quality scan.

---

## 🚀 Features
* **Real-time Processing:** Captures and processes video frames on the fly.
* **Bird’s Eye View:** Automatically corrects perspective for tilted documents.
* **Scan Effect:** Implements Gaussian thresholding to give the final output a professional "scanned" look.
* **Debug Stack:** Displays a visual step-by-step of the image processing pipeline.

## 🛠 Preprocessing Pipeline
The project follows a 6-step computer vision workflow:

1. **Capture:** Webcam initialization and raw frame acquisition.
2. **Resizing:** Images are scaled to a standard size to optimize processing speed and accuracy.
3. **Edge Detection:** * Conversion to grayscale.
    * **Canny Edge Detection** to identify boundaries.
    * A combination of **Dilation** and **Erosion** to close gaps and eliminate noise.
4. **Contour Extraction:** Algorithms identify the largest contour with exactly four corners.
5. **Perspective Transform:** * Points are reordered (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    * Applies a **Warp Perspective** to flatten the document.
6. **Adaptive Thresholding:** Applies a Gaussian threshold to clean up the text and mimic a scanned paper aesthetic.

## 💻 How to Run
### Prerequisites
Ensure you have Python installed along with the following libraries:
* **OpenCV:** `pip install opencv-python`
* **NumPy:** `pip install numpy`

### Execution
1. Download the project files.
2. Ensure your webcam is connected.
3. Run the script directly:
   ```bash
   python main.py
