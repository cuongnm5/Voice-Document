# TACOTRON 2 (December 2017)

## Original paper: https://arxiv.org/pdf/1712.05884.pdf

## 1. Tổng quan.

Paper giới thiệu Tacotron2, một mạng tổng hợp tiếng nói trực tiếp từ văn bản.

Hệ thống tổng hợp tiếng nói được cấu thành từ 2 phần. Phần 1 là một mạng Seq2Seq để map character embeddings với spectrogram trên thang đo Mel. Phần 2 là một model cải tiến từ WaveNet, tổng hợp ra âm thanh từ miền thời gian dạng sóng của spectrogram ở trên.

## 2. Giới thiệu.

Sinh ra lời nói tự nhiên từ văn bản (TTS) vốn là một thách thức dù đã có nhiều thập kỉ nghiên cứu nó. Rất nhiều kỹ thuật đã được áp dụng. Trong đó phương pháp " Concatenative
synthesis with unit selection, the process of stitching small units of pre-recorded waveforms together" hay tạm dịch là tổng hợp tiếng nói cho các từ và nối chúng lại thành 1 câu văn, đã dẫn đầu trong nhiều năm. Có nhiều phương pháp khác tuy nhiên âm thanh tạo ra không tự nhiên và không giống giọng con người.

Wavenet, một model tổng hợp từ miền thời gian dạng sóng, cung cấp audio chất lượng tốt và có thể so sánh với giọng thật của con người. Đầu vào của Wavenet gồm đặc trưng ngôn ngữ, predicted log fundamental frequency (F0), và phoneme durations. Điều này đòi hỏi kiến thức chuyên môn sâu về ngôn ngữ nói và phát âm.

Tacotron, kiến trúc Seq2Seq để tạo ra cường độ Spectrogram từ 1 câu, đơn giản hóa phương pháp tổng hợp tiếng nói truyền thống bằng cách bỏ đi ngôn ngữ học và acoustic feature, thay vào đó là một mạng neural được train từ data riêng biệt. Để âm thanh hóa phổ cường độ spectrograms, Tacotron sử dụng thuật toán Griffin-Lim cho phase estimation, sai đó biến đổi Fourier Short Time.

Ở paper này, nó mô tả cách tiếp cận trưc tiếp, kế thừa 2 cách tiếp cận tốt nhất từ trước: Mô hình Seq2Seq như Tacotron để gen ra Spectrograms theo thang Mel. Sau đó là bộ phát âm Wavenet được sửa đổi.

## 3. Kiến trúc model.

Hệ thống được đề xuất gồm 2 thành phần như được đề cập trong hình dưới: (1) một mạng recurrent sequence-to-sequence dự đoán feature với attention để dự đoán một chuỗi các khung mel spectrogram từ đầu vào là một đoạn ký tự. (2) một phiên bản cải tiến của WaveNet để tái tạo samples dạng sóng ở miền thời gian dựa trên khung mel spectrogram được dự đoán ở trên.

### 3.1. Biểu diễn feature trung gian.

Paper chọn cách biểu diễn âm thanh ở level thấp: melfrequency spectrograms. Cách biểu diễn này cho phép dễ tính toán từ time-domain waveforms, cho phép train thành 2 thành phần riêng biệt. Nó cũng mượt hơn dạng sóng và dễ train khi sử dụng bình phương loss vì STFT bất biến trên mỗi frame.

Melfrequency spectrograms có được bằng cách biến đổi phi tuyến lên chiều tần số của STFT. Cách tính này lấy cảm hứng từ việc tai người tiếp nhận âm thanh. Sau đó tóm tắt nội dung tần số với kích thước nhỏ hơn.

Sử dụng thang đó tần số ở level thấp có tác dụng nhấn rõ chi tiết của lời nói, tập trung vào độ rõ của giọng. Khi nhận mạnh vào tần số level cao sẽ bị dính tiếng ồn, noise do vậy không cần tập trung vào nó. Do tính chất này, thang đo mel được sử dụng làm đại diện cơ bản trong nhận dạng giọng nói trong nhiều thập kỷ.

Trong khi linear spectrograms bỏ qua thông tin pha, dẫn đến nhiều thông tin bị mất đi, tuy nhiên thuật toán như Griffin-Lim lại có khả năng ước tính những thông tin bị loại bỏ này và cho phép chuyển về miền thời gian bằng STFT. Các Mel spectrograms còn loại bỏ nhiều thông tin hơn nữa trong quá trình chuyển đổi, đưa ra một vấn đề chuyển đổi khó khăn. Tuy nhiên so với Linguistic và Acoustic feature trong WaveNet thì phổ mel là một biểu diễn âm thanh đơn giản, lowerlevel của tín hiệu âm thanh. Do đó, đơn giản hóa quá trình một mô hình WaveNet tương tự được điều chỉnh trên các phổ mel để tạo ra âm thanh, như một mạng neural vocoder. Thật vậy, chúng tôi sẽ chỉ ra rằng có thể tạo ra âm thanh chất lượng cao từ các phổ phổ bằng cách sử dụng kiến trúc WaveNet đã được sửa đổi.

### 3.2 Spectrogram Prediction Network.

![Image](images/tacotron0.jpeg)

#### Sơ lược.

Mạng predict ra spectrogram gồm 3 thành phần: Encoder, Decoder, Location Sensitive Attention.

Encoder: Câu từ được biểu diễn bởi character embedding 512 chiều. Sau đó đưa qua 3 layer tích chập, mỗi layer có shape 5x1 với 512 filters. Cuối cùng là một mạng LSTM 2 chiều chứa 256 units cho mỗi chiều.

Attention: Mục đích chính là để focus vào đặc trưng của cả các steps trước đó. Làm tăng tính nhất quán dữ liệu.

Decoder: Một mạng recurrent neural để predict melspectrogram từ 1 sequence đầu vào tại một thời điểm. Người ta cho rằng Pre-net hoạt động như information bottleneck thực sự cần thiết cho việc học attention. Pre-net output và attention context vector được nối lại và đi qua lớp LSTM với 1024 units. Kết nối này được chiếu tuyến tính để tìm ra target spectrogram frame.

Cuối cùng, predicted spectrogram qua 5 layer convolutional Post-net để dự đoán 1 phần dư, thêm vào prediction để improve tổng thể.

#### Chi tiết.

Giống như trong Tacotron, Mel Spectrograms được tính toán thông qua biến đổi Fourier ngắn (STFT) sử dụng frame size 50ms, hop frame 12.5 ms và hàm cửa sổ Hann. Tác giả cũng đã thử nghiệm với hop frame 5ms để match với tần số của input trong Original WaveNet nhưng sự gia tăng độ phân giải thời gian dẫn đến nhiều vấn đề về phát âm hơn, vì vậy không được sử dụng trong bài viết này.

Tác giả chuyển đổi cường độ STFT thành thang đo Mel bằng cách sử dụng Filter bank 80 channels có tần số từ 125Hz - 7.6kHz, sau đó nén dải động log. Trước khi nén, cường độ đầu ra của bộ lọc được cắt ở giá trị tối thiểu 0,01 để giới hạn phạm vi động trong miền logarit.

The network is composed of an encoder and a decoder with attention. The encoder converts a character sequence into a hidden feature
representation which the decoder consumes to predict a spectrogram.
Input characters are represented using a learned 512-dimensional
character embedding, which are passed through a stack of 3 convolutional layers each containing 512 filters with shape 5 × 1, i.e., where
each filter spans 5 characters, followed by batch normalization  
and ReLU activations. As in Tacotron, these convolutional layers
model longer-term context (e.g., N-grams) in the input character
sequence. The output of the final convolutional layer is passed into a
single bi-directional LSTM layer containing 512 units (256
in each direction) to generate the encoded features.

The encoder output is consumed by an attention network which
summarizes the full encoded sequence as a fixed-length context vector
for each decoder output step. We use the location-sensitive attention
from which extends the additive attention mechanism to
use cumulative attention weights from previous decoder time steps
as an additional feature. This encourages the model to move forward
consistently through the input, mitigating potential failure modes
where some subsequences are repeated or ignored by the decoder.
Attention probabilities are computed after projecting inputs and location features to 128-dimensional hidden representations. 

Location features are computed using 32 1-D convolution filters of length 31.
The decoder is an autoregressive recurrent neural network which
predicts a mel spectrogram from the encoded input sequence one
frame at a time. The prediction from the previous time step is first
passed through a small pre-net containing 2 fully connected layers
of 256 hidden ReLU units. We found that the pre-net acting as an
information bottleneck was essential for learning attention. The prenet output and attention context vector are concatenated and passed
through a stack of 2 uni-directional LSTM layers with 1024 units.

The concatenation of the LSTM output and the attention context
vector is projected through a linear transform to predict the target
spectrogram frame. Finally, the predicted mel spectrogram is passed
through a 5-layer convolutional post-net which predicts a residual
to add to the prediction to improve the overall reconstruction. Each
post-net layer is comprised of 512 filters with shape 5 × 1 with batch
normalization, followed by tanh activations on all but the final layer.
We minimize the summed mean squared error (MSE) from before
and after the post-net to aid convergence. We also experimented
with a log-likelihood loss by modeling the output distribution with
a Mixture Density Network [23, 24] to avoid assuming a constant
variance over time, but found that these were more difficult to train
and they did not lead to better sounding samples.

In parallel to spectrogram frame prediction, the concatenation of
decoder LSTM output and the attention context is projected down
to a scalar and passed through a sigmoid activation to predict the
probability that the output sequence has completed. This “stop token”
prediction is used during inference to allow the model to dynamically
determine when to terminate generation instead of always generating
for a fixed duration. Specifically, generation completes at the first
frame for which this probability exceeds a threshold of 0.5.

The convolutional layers in the network are regularized using
dropout with probability 0.5, and LSTM layers are regularized
using zoneout with probability 0.1. In order to introduce output
variation at inference time, dropout with probability 0.5 is applied
only to layers in the pre-net of the autoregressive decoder.
In contrast to the original Tacotron, our model uses simpler building blocks, using vanilla LSTM and convolutional layers in the encoder and decoder instead of “CBHG” stacks and GRU recurrent
layers. We do not use a “reduction factor”, i.e., each decoder step
corresponds to a single spectrogram frame.
