## **1. Đặc trưng đầu vào (Feature Front-end)**

* **80-dimensional filter-bank features**:

  * Âm thanh thô (waveform) => tính phổ năng lượng qua FFT.
  * Dùng **FFT 25ms**, dịch 10ms => thu được chuỗi vector 80 chiều/mỗi frame.

* **Data augmentation**:

  * **Speed Perturb**: thay đổi tốc độ phát (0.9x, 1.0x, 1.1x).
  * **SpecAugment**: làm mờ hoặc xóa vùng trong **trục thời gian hoặc tần số** của spectrogram.

---

## **2. Biểu diễn đầu ra (Output Vocabulary)**

* Sử dụng **Byte-Pair Encoding (BPE)**, vocab size = **5000 tokens**.
* Mỗi token → vector embedding **256 chiều**.

---

## **3. Quá trình training (Full-context training trước)**

* **Framework**: WeNet 2.0
* **Hardware**: 8 × NVIDIA H100 GPUs
* **Mixed precision**: dùng FP16 để tăng tốc và giảm bộ nhớ.
* **Training schedule**:

  * **Small dataset**: 200 epochs.
  * **Large dataset**: 400,000 steps.
* **Optimizer**: **Adam**.
* **Scheduler**: **Noam warm-up** (giống Transformer paper)

  * LR tăng dần đến **1e-3** sau 15k steps (small) hoặc 25k steps (large).
  * LR giảm dần theo 1/√step.

---

## **4. Fine-tuning (Limited-context / Chunk training)**

* Sau khi có **full-context model pre-trained**, fine-tune thành **masked batch limited context models**:

  * Số epoch: **50 (small)**, **100k steps (large)**.
  * Reset learning rate nhỏ hơn (**1e-5**) để tránh làm hỏng trọng số đã học.

---

## **5. Checkpoint Averaging**

* Sau training, không lấy 1 checkpoint cuối cùng → mà lấy **trung bình trọng số**:

  * Small: trung bình **50 checkpoint cuối**.
  * Large: trung bình **10 checkpoint cuối**.

---

## **6. Dynamic Context Training**

* Khi train limited-context, không giữ chunk cố định mà **random thay đổi kích thước và vị trí context**:
  * Thay đổi các tham số: **latt, c, r** (tức left context, center chunk, right context).
---