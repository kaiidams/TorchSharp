using System;
using Xunit;

using TorchSharp;

namespace TorchSharp
{
    public class TestTorchAudio
    {
        private torch.Tensor make_waveform()
        {
            var fc = 220;
            var fm = 440;
            var beta = 1;
            var t = torch.linspace(0, 5, 16000 * 5);
            var waveform = 1.0 * torch.sin(2 * Math.PI * t * fc + beta * torch.sin(2 * Math.PI * fm * t));
            return waveform.unsqueeze(0);
        }

        [Fact]
        public void FunctionalSpectrogram()
        {
            var waveform = make_waveform();

            var spectrogram = torchaudio.functional.spectrogram(
                waveform: waveform,
                pad: 3,
                window: torch.hann_window(400),
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                power: 2.0f,
                normalized: true);

            Assert.Equal(new long[] { 1, 257, 501 }, spectrogram.shape);
            var mean_square = torch.mean(torch.square(spectrogram)).item<float>();
            Assert.InRange(mean_square - 50.7892f, -1e-2f, 1e-2f);
        }

        [Fact]
        public void FunctionalInverseSpectrogram()
        {
            var waveform = make_waveform();

            var spectrogram = torchaudio.functional.spectrogram(
                waveform: waveform,
                pad: 3,
                window: torch.hann_window(400),
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                power: null,
                normalized: false);

            var inversed_waveform = torchaudio.functional.inverse_spectrogram(
                spectrogram: spectrogram,
                length: null,
                pad: 3,
                window: torch.hann_window(400),
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                normalized: false);

            var mse = torch.mean(torch.square(waveform - inversed_waveform)).item<float>();
            Assert.InRange(mse, 0, 0.1f);
        }

        [Fact]
        public void TransformsSpectrogram()
        {
            var transform = torchaudio.transforms.Spectrogram();
            var waveform = make_waveform();
            var spectrogram = transform.forward(waveform);
            var expected = torchaudio.functional.spectrogram(
                waveform: waveform,
                pad: 0,
                window: torch.hann_window(400),
                n_fft: 400,
                hop_length: 200,
                win_length: 400,
                power: 2.0,
                normalized: false);
            var mse = torch.mean(torch.square(spectrogram - expected)).item<float>();
            Assert.InRange(mse, 0f, 1e-10f);
        }

        [Fact]
        public void TransformsInverseSpectrogram()
        {
            var transform = torchaudio.transforms.InverseSpectrogram();
            var spectrogram = torchaudio.functional.spectrogram(
                waveform: make_waveform(),
                pad: 0,
                window: torch.hann_window(400),
                n_fft: 400,
                hop_length: 200,
                win_length: 400,
                power: null,
                normalized: false);
            var waveform = transform.forward(spectrogram);
            var expected = torchaudio.functional.inverse_spectrogram(
                spectrogram: spectrogram,
                length: null,
                pad: 0,
                window: torch.hann_window(400),
                n_fft: 400,
                hop_length: 200,
                win_length: 400,
                normalized: false);
            var mse = torch.mean(torch.square(waveform - expected)).item<float>();
            Assert.InRange(mse, 0f, 1e-10f);
        }

        [Fact]
        public void TestFunctionalResampleIdent()
        {
            var waveform = make_waveform();
            var resampled_waveform = torchaudio.functional.resample(waveform, 16000, 16000);
            Assert.Equal(waveform, resampled_waveform);
        }

        [Fact]
        public void FunctionalResampleUpSample()
        {
            var waveform = make_waveform();
            var resampled_waveform = torchaudio.functional.resample(waveform, 16000, 24000);
            var x = waveform.reshape(1, -1, 2).mean(dimensions: new long[] { -1 });
            var y = resampled_waveform.reshape(1, -1, 3).mean(dimensions: new long[] { -1 });
            var mse = torch.mean(torch.square(x - y)).item<float>();
            Assert.True(mse < 1e-2);
        }

        [Fact]
        public void FunctionalResampleDownSample()
        {
            var waveform = make_waveform();
            var resampled_waveform = torchaudio.functional.resample(waveform, 16000, 8000);
            var x = waveform.reshape(1, -1, 2).mean(dimensions: new long[] { -1 });
            var y = resampled_waveform;
            var mse = torch.mean(torch.square(x - y)).item<float>();
            Assert.True(mse < 1e-2);
        }

        [Fact]
        public void TransformsResampleDownSample()
        {
            var waveform = make_waveform();
            var transform = torchaudio.transforms.Resample(16000, 8000, device: waveform.device, dtype: waveform.dtype);
            var resampled_waveform = transform.forward(waveform);
            var x = waveform.reshape(1, -1, 2).mean(dimensions: new long[] { -1 });
            var y = resampled_waveform;
            var mse = torch.mean(torch.square(x - y)).item<float>();
            Assert.True(mse < 1e-2);
        }
    }
}