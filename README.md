# Jetracer AI Racing Example Demo
Đây là repo mẫu cho cuộc thi jetracer AI Racing tổ chức vào kì Summer 2026, được build up từ https://github.com/waveshare/jetracer_ros

Repo này được test trong một điều kiện nhất định, do đó việc khả năng code có thể chạy trên máy khác sẽ khá thấp, các bạn chỉ nên tham khảo về cấu trúc thay vì các thuật toán và config của repo này
## 1.Một số thông tin cần biết
Một số phiên bản phần mềm đang sử dụng trong Jetracer:

| Thành phần | Phiên bản |
| :--- | :--- |
| **Jetpack** | 4.5.1 |
| **Python 3** | 3.6.9 |
| **Python 2** | 2.7.1 |
| **ROS 1** | Melodic |

**Các bạn lưu ý:** Riêng về ROS sẽ sử dụng các phiên bản python2, do có các script sử dụng python3 có sử dụng thông tin từ ROS thì hầu như sẽ không thực hiện được.
## 2.Cấu Trúc Cần Lưu Ý

- `launch/`: file khởi chạy ROS. Demo chính dùng `step_drive.launch`.
- `config/`: tham số YAML cho lane-following, maneuver, state machine và YOLO.
- `scripts/state_machine_controller_node.py`: state machine chính, nhận lệnh
  từng bước và sở hữu `/cmd_vel`.
- `scripts/lane_following_node.py`: xử lý lane và publish Twist nội bộ.
- `scripts/maneuver_node.py`: xử lý đi thẳng qua giao lộ, rẽ trái, rẽ phải.
- `scripts/trt_udp_detector.py`: detector TensorRT chạy Python 3.
- `scripts/yolo_bridge_node.py`: bridge detection từ UDP sang ROS topic.
- `scripts/lane_following/`: code xử lý ảnh tách riêng khỏi ROS node.
- `src/`: driver C++ cho JetRacer.
- `maps/`: script và dữ liệu liên quan tới lưu bản đồ.

## 3.Luồng Chính

Pipeline nên chia thành ba phần:

1. **Perception**: đọc camera, nhận diện lane, biển báo, đèn giao thông.
2. **State machine**: quyết định bước tiếp theo dựa trên lệnh và perception.
3. **Controller**: gửi lệnh cuối cùng xuống xe qua `/cmd_vel`.

Demo hiện đi theo hướng step-by-step: gửi một lệnh, xe thực hiện xong thì dừng và
quay về `IDLE`. Cách này dễ debug hơn chạy tự động liên tục.

## 4.Cách Chạy Nhanh

```bash
catkin_make
source devel/setup.bash
roslaunch jetracer step_drive.launch
```

Gửi lệnh thử:

```bash
rostopic pub /sm/command std_msgs/String "data: 'turn_left'"
```

Theo dõi state:

```bash
rostopic echo /sm/state
```

## 5. Quy trình để xây dựng một pipeline
Đây là kinh nghiệm riêng của người đã xây dựng repo này, các bạn có thể tham khảo hoặc không, mình và cả Ban Tổ chức KHÔNG BẮT BUỘC các bạn làm theo
- Bước 1: Xây dựng sơ lược function mà cần xây dựng: input là gì, output là gì, cần sử dụng gì cho function này
- Bước 2: Viết script python prototype (đảm bảo luôn có log để có thể điều chỉnh dễ dàng hơn)
- Bước 3: Thử nghiệm và sửa lỗi, kiểm tra tính khả thi
- Bước 4: Port sang ROS và sửa các config còn sai sót.

**Q**: Liệu chúng ta có thể viết ROS từ đầu không?

**A**: Hoàn toàn có thể, tuy nhiên đôi lúc sẽ hơi khó để kiểm tra và sửa các lỗi, việc này ở 1 script python nhỏ làm tốt hơn. Tuy nhiên, nếu bạn hoàn thành thì sẽ đỡ tốn thời gian port sang ROS và cũng đảm bảo tỉ lệ chính xác cao hơn. Lựa chọn là ở các bạn.
## 6.Model AI 
Ở phần này, Ban tổ chức sẽ nói sơ lược một số quy trình cần cho AI để các bạn **THAM KHẢO** (KHÔNG BẮT BUỘC CÁC BẠN LÀM THEO):
- Bước 1: Thu thập dữ liệu từ sa bàn mẫu
- Bước 2: Training model
- Bước 3: Kiểm tra, đánh giá model. Nếu model đạt chuẩn thì tiếp tục bước 4, không thì hãy thử kiểm tra xem data của mình đã ổn chưa và cấu trúc của model AI mình chọn đã đáp ứng bài toán tốt chưa
- Bước 4: Port sang TensorRT

**Q**: Không port sang TensortRT được không ?

**A**: TensortRT cũng chỉ là cách tối ưu, bạn có thể sử dụng bất kì giải pháp nào khác nếu thỏa mãn bài toán.

## 7.Kết nối Jetson
- Bước 1: Đảm bảo Jetson của bạn và thiết bị của bạn cùng Wifi

- Bước 2: SSH vào Jetson bằng lệnh ``ssh jetson@<ip của jetson>`` hoặc kết nối vào jupyterlab của jetson bằng cách sử dụng trình duyệt, vào  ``<ip của jetson>:8888`` 

Lưu ý điều này: **WIFI trường chặn SSH**


## 8.Một số vấn đề ngoài lề
- Tuy là xe sử dụng GPU của nhà Nvidia nhưng đây vẫn là Edge device, do đó khá yếu, việc bạn ứng dụng trên laptop của bạn và trên xe là khác nhau.

- Ram trên Jetson là unified ram tức CPU và GPU dùng chung
