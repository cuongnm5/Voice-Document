# TACOTRON (March 2017)

## Original paper: https://arxiv.org/pdf/1703.10135.pdf

## 1. Tổng quan.

Xây dựng một hệ thống tổng hợp tiếng nói (TTS) chuyên dụng bao gồm nhiều stages, như là tiền xử lý, phân tích văn bản, mô hình ngữ âm học, mô đun tổng hợp lên âm thanh. Xây dựng những thành phần này thường yêu cầu nền tảng kinh nghiệm chuyên sâu. Trong paper này, tác giả giới thiệu Tacotron, một end-to-end generative text-to-speech model tổng hợp tiếng nói trực tiếp từ ký tự. Khi có cặp <text, audio>, mô hình có thể huấn luyện hoàn chỉnh from scratch với khởi đầu random. Tác giả giới thiệu một số kỹ thuật chính để tạo nên một framework Seq2Seq hoạt động hiệu quả với challenging task này. Tacotron đạt được 3.82/5 MOS trên tiếng Anh (US English), vượt trội hơn hẳn so với một hệ thống production parametric về độ tự nhiên của giọng. Hơn nữa, vì Tacotron tạo ra giọng nói ở frame-level, nó nhanh hơn đáng kể so với sample-level của phương pháp autoregressive.

## 2. Giới thiệu.

Modern TTS pipelines rất phức tạp (Taylor, 2009). Ví dụ, thông thường tham số thống kê của TTS có trước khi văn bản trích xuất các features ngôn ngữ học khác nhau, mô hình duration, mô hình dự đoán tính năng âm thanh và một bộ phát âm dựa trên xử lý tín hiệu phức tạp (Zen
et al., 2009; Agiomyrgiannakis, 2015). Các thành phần này dựa trên kinh nghiệm chuyên sâu và rất khó để thiết kế. Họ cũng huấn luyện các thành phần riêng biệt, vậy nên lỗi xảy ra ở mỗi thành phần có thể tạo ra lỗi lớn. 
Do đó, sự phức tạp của các thiết kế TTS hiện đại dẫn đến những nỗ lực kỹ thuật đáng kể khi xây dựng một hệ thống mới.

Có rất nhiều lợi thế của hệ thống TTS end2end tích hợp có thể được huấn luyện bằng cặp <text, audio> với sự góp mặt ít nhất bởi con người. Đầu tiên, một hệ thống như vậy làm giảm yêu cầu về kỹ thuật tính năng tốn kém mà có thể liên quan đến heuristics và brittle design choices. Thứ hai, nó dễ dàng điều chỉnh các thuộc tính khác nhau hơn, chẳng hạn như speaker hoặc ngôn ngữ hoặc các feature cấp cao như sentiment.

Điều này bởi vì việc điều chỉnh (conditioning) có thể xảy ra ngay từ đầu của mô hình chứ không chỉ trên một số thành phần nhất định. Tương tự, việc thích ứng với dữ liệu mới cũng có thể dễ dàng hơn. Cuối cùng, một mô hình duy nhất có khả năng robust mạnh hơn là mô hình nhiều tầng mà ở đó mỗi errors thành phần có thể bị gộp lại thành một big errors. Những lợi thế này ngụ ý rằng một mô hình đầu cuối có thể cho phép chúng ta đào tạo trên một lượng lớn dữ liệu phong phú, gồm cả biểu cảm và tiếng ồn trong thế giới thực.



TTS is a large-scale inverse problem: a highly compressed source (text) is “decompressed” into
audio. Since the same text can correspond to different pronunciations or speaking styles, this is a
particularly difficult learning task for an end-to-end model: it must cope with large variations at the signal level for a given input. Moreover, unlike end-to-end speech recognition (Chan et al., 2016)
