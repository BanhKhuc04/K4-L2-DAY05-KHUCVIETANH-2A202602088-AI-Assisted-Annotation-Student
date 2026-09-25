# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Khúc Việt Anh

Công cụ gán nhãn đã dùng: CVAT (chạy Docker trên máy cá nhân)

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Vì ảnh lấy liên tiếp từ cùng một video, các khung hình cách nhau vài giây gần như giống hệt nhau về
xe, góc quay và ánh sáng. Nếu chia ngẫu nhiên, một ảnh gần trùng với ảnh trong tập test rất dễ lọt
vào pool để huấn luyện, khiến mô hình gần như "thuộc lòng" đúng cảnh đó khi được đánh giá lại trên
test — đây là rò rỉ dữ liệu (data leakage). Hướng lệch sẽ luôn là **số đo bị đội lên giả tạo** (AP50,
recall cao hơn thực tế), khiến ta lầm tưởng mô hình tổng quát tốt trong khi thực chất chỉ đang nhận
diện lại cảnh nó đã thấy. Chia theo trục thời gian kèm vùng đệm giữ pool và test đủ xa nhau về mặt
thời gian, giảm khả năng trùng cảnh giữa hai tập.

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi đầu lạnh không khớp nhãn tham chiếu ở
những loại xe nào. Độ phủ (recall) theo kích thước xe cho thấy điều gì? Một trường hợp nào cần
người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

Trên 4 ảnh mẫu (frame_0050, 0150, 0250, 0350), mô hình cold start có hai kiểu lỗi lặp lại: (1) khung
đỏ (False Positive) — chủ yếu là vệt sáng đèn xe hoặc ánh phản chiếu trên đường bị nhận nhầm thành
xe (2 FP ở gần như mọi ảnh mẫu); (2) khung vàng bỏ sót (False Negative) — tập trung ở các xe nhỏ,
xa, nằm phía trên khung hình, nơi chỉ còn thấy đốm đèn mờ. Bảng recall theo kích thước xác nhận rõ
điều này: xe nhỏ chỉ đạt recall 0.182 (66 box tham chiếu), trong khi xe cỡ vừa và lớn đạt 0.547 và
0.561 — mô hình cold start gần như "mù" với xe nhỏ/xa. Trước khi kết luận hoàn toàn do mô hình yếu,
cần người rà lại các box tham chiếu rất nhỏ (sát ngưỡng bỏ qua 16px): ở khoảng cách xa như vậy, ngay
cả người cũng khó chắc chắn một đốm sáng là xe thật hay chỉ là vệt sáng — một phần recall thấp có
thể đến từ nhãn tham chiếu biên giới, không hoàn toàn là lỗi mô hình.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Công thức kết hợp ba tín hiệu: **U** — mức mô hình không chắc chắn (trung bình độ tin cậy thấp nhất
trong ảnh); **A** — tỷ lệ box có độ tin cậy rất thấp (dưới 0.15) trên tổng box nghi ngờ (dưới 0.50);
**D** — khoảng cách thời gian tới ảnh đã chọn gần nhất, giúp tránh chọn hai ảnh liền kề gần như giống
hệt nhau. `MIN_GAP_S` đặt khoảng cách tối thiểu giữa các ảnh được chọn, nên dù một ảnh có U, A rất
cao, nếu nó quá gần một ảnh đã chọn thì vẫn bị loại — đúng như trường hợp frame_0372.jpg (điểm
0.9101, hạng 6) bị loại vì chỉ cách frame_0369.jpg đã chọn 1.2 giây (chi tiết ở `SELECTION.md`).

Ba frame frame_0270.jpg, frame_0331.jpg, frame_0369.jpg (dẫn chứng đầy đủ trong `SELECTION.md`) cho
thấy điểm cao có thể đến từ hai nguyên nhân khác nhau: frame_0331.jpg điểm cao vì model **nhận nhầm
nhiều** (6 box bị xóa — cao nhất lô), trong khi frame_0369.jpg điểm cao vì model **quá tự tin nhưng
bỏ sót nhiều** (14/28 box phải thêm mới, dù 14 box đề xuất đều đúng 100%). Điểm bất định **không**
chứng minh ảnh đó chắc chắn sẽ cải thiện mô hình khi học — nó chỉ cho biết mô hình *hiện tại* chưa tự
tin ở đó. Bằng chứng rõ nhất: dù toàn bộ 12 ảnh vòng 1 đều nằm trong nhóm uncertainty cao nhất pool,
AP50 sau khi học lại **giảm** (0.771 → 0.730, xem Mục 4) — cho thấy giảm bất định của model không
đồng nghĩa cải thiện số đo tổng thể trên tập test.

## 4. Các vòng học chủ động (active learning)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 297 | 0.730 | -0.041 | 0.991 | 0.268 | 0.422 | 0.000 | 0.267 | 0.707 |

Với mỗi vòng, trình bày: mức độ bạn đã sửa nhãn gợi ý; AP50 thay đổi bao nhiêu so với khởi đầu lạnh
và so với vòng trước; nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

**Vòng 1** — model đề xuất 169 box trên 12 ảnh, sau khi rà còn 297 box: **accepted 127**
(giữ nguyên), **edited 23** (sửa lệch/sai cỡ), **deleted 19** (model nhận nhầm), **added 147** (model
bỏ sót) — tỷ lệ chấp nhận nguyên trạng (accept rate) 75.15%. Số box thêm mới (147) gần bằng toàn bộ
số box đề xuất ban đầu (169), cho thấy pre-label bỏ sót rất nhiều trên lô ảnh có độ bất định cao này.

So với cold start, AP50 giảm nhẹ (0.771 → 0.730, Δ = -0.041), nhưng các chỉ số phía sau thay đổi rất
mạnh và trái chiều nhau: **precision tăng vọt** (0.925 → 0.991, FP giảm từ 16 xuống chỉ còn 1), trong
khi **recall sụt mạnh** (0.489 → 0.268, FN tăng từ 206 lên 295). Theo kích thước xe: recall xe nhỏ
**mất hoàn toàn** (0.182 → 0.000), xe vừa **xấu đi** (0.547 → 0.267), riêng xe lớn **tốt lên**
(0.561 → 0.707). Nói cách khác, sau khi học từ 12 ảnh, mô hình trở nên rất "cẩn trọng" — gần như
không còn báo nhầm — nhưng đổi lại bỏ sót phần lớn xe nhỏ/vừa, chỉ còn nhạy với xe lớn/gần.

Từ ảnh `outputs/compare_round1.jpg`, một ca cụ thể: **frame_0250** — cold start có TP 6, FP 2, FN 9;
sau vòng 1 vẫn TP 6, FN 9 (không đổi) nhưng **FP giảm về 0** — đúng với xu hướng chung: mô hình sau
train không tạo thêm nhầm lẫn mới, nhưng cũng không bắt được thêm xe nào đã bị bỏ sót từ trước.

Dùng ba nguồn để phân biệt rõ từng bước:
- **BLIND_SCAN.md** (quan sát độc lập, chưa xem nhãn AI): trên frame_0312.jpg, tôi đếm bằng mắt 23
  xe và ghi trước hai nghi vấn — xe bị cắt một phần ở mép dưới khung hình, và hai xe đèn đỏ đi sát
  nhau có thể bị AI gộp/chồng lấn.
- **REVIEW_LOG.csv** (lỗi pre-label thực tế đã sửa trên CVAT): ghi nhận model nhận nhầm vật thể
  thành xe tải và xe buýt (xóa khung) và một trường hợp 2 xe bị gộp chung 1 khung (sửa khung) — đúng
  khớp với nghi vấn "chồng lấn" đã ghi trước trong BLIND_SCAN.md.
- **round1_diff.md** (kết quả định lượng sau khi sửa): riêng frame_0312.jpg có accepted 9, edited 2,
  deleted 2, added 10 trên 13 box đề xuất ban đầu — xác nhận bằng số cả hai nghi vấn quan sát độc
  lập đều là vấn đề thật, không phải suy diễn.

Một ca khó theo GUIDELINE_LABEL.md: **frame_0331.jpg** có tới 6 box bị xóa — cao nhất trong 12 ảnh —
nhiều khả năng là các vệt đèn phản chiếu bị model nhận nhầm thành xe riêng biệt; xử lý đúng theo quy
tắc là xóa hẳn các khung này thay vì chỉ sửa nhỏ vị trí, vì đó không phải lỗi lệch khung mà là lỗi
nhận diện sai đối tượng (False Positive) hoàn toàn.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

So với cold start, vòng 1 không cải thiện AP50 (giảm nhẹ 0.041 điểm) nhưng đổi lại gần như loại bỏ
hoàn toàn báo nhầm (precision 0.991) — đánh đổi lấy việc bỏ sót rất nhiều xe nhỏ/vừa. Đây chưa phải
kết quả xấu hoàn toàn (độ chính xác cao rất có giá trị trong một số tình huống), nhưng recall sụt
mạnh là dấu hiệu đáng lo, nhiều khả năng do chỉ train trên 12 ảnh với 50 epoch — mô hình học quá
khớp (overfit) vào đặc điểm của đúng 12 ảnh đó thay vì tổng quát hoá. Tôi đề xuất **tiếp tục thêm
vòng 2** nhưng cần điều chỉnh cách train (giảm epoch hoặc thêm ảnh đa dạng hơn về xe nhỏ/xa) thay vì
chỉ lặp lại y hệt cách làm vòng 1.

Hai ca đề xuất cho vòng sau (từ `outputs/selection_round2.csv`):
- **frame_0031.jpg** (hạng 1, score 0.8892, U=0.7784, D=1.0) — điểm bất định cao nhất pool còn lại,
  và cách xa mọi ảnh đã chọn (D=1.0) nên gần như chắc chắn không trùng nội dung với 12 ảnh vòng 1.
- **frame_0171.jpg** (hạng 8, score 0.7165, D=0.44 — thấp nhất trong lô vòng 2) — nêu ra như một ca
  **cảnh báo rủi ro gần trùng**: D thấp nghĩa là ảnh này khá gần về thời gian với một ảnh khác đã
  được chọn cùng lô, nên giá trị thông tin thêm có thể thấp hơn số điểm gợi ý; nên xem kỹ ảnh trước
  khi quyết định có đưa vào rà hay bỏ qua để tiết kiệm công.

Chi phí rà nhãn cho vòng 2 gần tương đương vòng 1 (12 ảnh, ước lượng cùng khối lượng thời gian ~70
phút theo lịch trình bài). Giới hạn của tập test (chỉ 20 ảnh, bỏ qua box dưới 16px, nhãn tham chiếu
do chính mô hình tạo ra chứ chưa được người rà) khiến các con số — đặc biệt recall xe nhỏ, vốn chỉ
dựa trên 66 box tham chiếu — dễ dao động mạnh chỉ vì vài ảnh, và không nên coi AP50 là thước đo tuyệt
đối duy nhất. Nếu AP50 tiếp tục giảm ở vòng sau, trước khi train thêm tôi sẽ kiểm tra: (1) các box
mới thêm (added) có được vẽ đúng kích thước/quy tắc theo GUIDELINE_LABEL.md không, đặc biệt với xe
nhỏ; (2) có đang train quá nhiều epoch so với lượng ảnh ít hay không (dấu hiệu overfit); (3) xem lại
`compare_round*.jpg` để xác định lỗi tập trung ở nhóm xe nào trước khi kết luận cần thêm dữ liệu hay
cần đổi cách train.
