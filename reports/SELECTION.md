# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

Nếu chỉ đủ ngân sách rà 5 ảnh, tôi ưu tiên đúng 5 ảnh điểm cao nhất trong bảng: **frame_0182.jpg**
(hạng 1, t=72.8s, score=0.9591), **frame_0369.jpg** (hạng 2, t=147.6s, score=0.9324),
**frame_0380.jpg** (hạng 3, t=152.0s, score=0.9170), **frame_0326.jpg** (hạng 4, t=130.4s,
score=0.9155), **frame_0331.jpg** (hạng 5, t=132.4s, score=0.9154). Cả 5 ảnh đều có U ≈ 0.83–0.93
(model gần như không chắc chắn) và n_ambiguous rất cao (15–18 box có độ tin cậy thấp), nên đây là
nhóm mang lại nhiều thông tin nhất trên mỗi ảnh rà.

Một quyết định liên quan đến ảnh gần trùng: **frame_0372.jpg** (hạng 6, t=148.8s, score=0.9101)
gần như ngang điểm với frame_0369.jpg (hạng 2), nhưng cách nhau chỉ 1.2 giây trong video nên gần
như chắc chắn quay cùng một nhóm xe. Công cụ chọn (yếu tố D, khoảng cách thời gian tối thiểu
`MIN_GAP_S`) đã tự loại frame_0372.jpg (`selected=False`) để tránh lãng phí công rà trên nội dung
gần trùng; tôi đồng ý với lựa chọn này và không đưa frame_0372.jpg vào danh sách 5 ảnh ưu tiên.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

- **frame_0270.jpg** (score=0.8878, 13 box gợi ý). Theo `outputs/round1_diff.json`: accepted 11,
  edited 2, deleted 0, added 10 → model không có box sai (deleted=0) nhưng bỏ sót gần bằng số đã
  đúng (10 box thêm mới trên 13 box đề xuất ban đầu).
- **frame_0331.jpg** (score=0.9154, 20 box gợi ý — nhiều nhất trong lô). Theo `round1_diff.json`:
  accepted 13, edited 1, **deleted 6** (cao nhất trong 12 ảnh), added 11 → đây là ảnh model nhận
  nhầm nhiều nhất (có thể do ánh đèn phản chiếu bị đếm thành xe).
- **frame_0369.jpg** (score=0.9324, 14 box gợi ý). Theo `round1_diff.json`: accepted 14, edited 0,
  deleted 0 → toàn bộ box AI đề xuất đều đúng, nhưng vẫn phải thêm 14 box mới (bỏ sót đúng bằng số
  đã phát hiện được), khớp với điểm uncertainty rất cao (U=0.9315) của ảnh này.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

frame_0372.jpg (điểm cao, hạng 6, 0.9101) không được chọn vì lý do đã nêu ở trên (gần trùng thời
gian với frame_0369.jpg đã chọn). Ngược lại, ở nửa cuối bảng có nhiều frame điểm thấp (0.57–0.65)
vẫn đáng xem nhanh dù không nằm trong ngân sách chính: các ảnh này thường có `n_boxes` và
`n_ambiguous` nhỏ (ví dụ dưới 20 và dưới 5), nghĩa là ít xe hoặc model khá tự tin — rủi ro bỏ sót
thấp hơn, nên hợp lý khi xếp cuối danh sách ưu tiên.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:

Điểm số (Score) chỉ đo mức **model không chắc chắn** ở ảnh đó (dựa trên U, A) và mức **đa dạng thời
gian** so với ảnh đã chọn (D) — nó không đo nội dung ảnh có khó gán nhãn hay không, và càng không
đảm bảo việc rà/sửa ảnh đó sẽ làm AP50 tăng. Thực tế ở vòng 1: dù 12 ảnh được chọn đều là nhóm điểm
uncertainty cao nhất trong pool, AP50 sau fine-tune lại **giảm nhẹ** (0.771 → 0.730, xem
`reports/REPORT.md` Mục 4) — minh chứng rõ cho việc điểm chọn ảnh cao không đồng nghĩa mô hình sẽ
tốt lên sau khi học từ ảnh đó.
