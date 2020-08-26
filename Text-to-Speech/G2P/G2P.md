# Sequence-to-Sequence Neural Net Models for Grapheme-to-Phoneme Conversion (20 Aug 2015)
## Original paper: https://arxiv.org/pdf/1506.00196.pdf

## 1. Tổng quan
Chuyển đổi hình vị - âm vị (Grapheme-to-Phoneme, G2P) là bài toán quan trọng liên quan đến vấn đề xử lý ngôn ngữ, xử lý tiếng nói. Mục đích của G2P là dự đoán chính xác cách phát âm của một từ mới trong văn bản đầu vào chỉ dựa trên phân tích chính tả.

Các phương pháp dịch Sequence-to-Sequence dựa trên generation với mô hình ngôn ngữ có điều kiện đã cho thấy kết quả khả quan trong nhiều tasks. Trong dịch máy, các mô hình được điều chỉnh dựa trên các từ cơ bản để tạo nên target-language text và còn được sử dụng trong việc tạo văn bản phụ đề. 
Cách tiếp cận trước đây và bây giờ chủ yếu tập trung vào large vocabulary tasks và đo lường chất lượng theo BLEU. Trong paper này, tác giả khám phá ra rằng khả năng ứng dụng của các mô hình vào Grapheme-to-Phoneme có sự khác biệt về chất lượng. Ở đây, các từ vựng bên phía đầu vào và đầu ra là các mô hình n-gram nhỏ, đơn giản nhưng hiệu quả. Credit chỉ được đưa ra khi đầu ra là kết quả chính xác. Tác giả thấy rằng có thể cải thiện đáng kể state-of-the-art với LSTM.

## 2. Giới thiệu

Trong các nghiên cứu gần đây về Seq2Seq Translation, người ta đã chứng minh rằng các mạng side-conditioned neural có thể đạt hiệu quả cho cả dịch máy và chú thích hình ảnh. Việc sử dụng mô hình side-conditioned language rất hấp dẫn vì nó tính đơn giản và hiệu suất rõ ràng.

Trong các tasks được nghiên cứu trước đây, kích thước của từ vựng đầu vào là rất lớn và số liệu thống kê cho rất nhiều từ được ước lượng không đồng đều. Để giảm bớt vấn đề này, các cách tiếp cận dựa trên neural network sử dụng các cách biểu thị không gian liên tục của từ (continuous-space representations of words), trong đó từ xuất hiện trong bối cảnh tương tự nhau có xu hướng gần nhau hơn trong không gian biểu diễn. Do đó dữ liệu có lợi cho một từ trong ngữ cảnh cụ thể khiến mô hình tổng quát hóa được các từ tương tự trong các bối cảnh tương tự. Tóm lại, để giải quyết vấn đề số liệu thống kê bất ổn định, mạng thần kinh như RNN và LSTM tỏ ra rất hiệu quả.