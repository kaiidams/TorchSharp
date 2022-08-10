// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/c15eee23964098f88ab0afe25a8d5cd9d728af54/torchaudio/models/wavernn.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

#nullable enable
namespace TorchSharp.Modules
{
    // ResNet block based on *Efficient Neural Audio Synthesis* [:footcite:`kalchbrenner2018efficient`].
    // 
    //     Args:
    //         n_freq: the number of bins in a spectrogram. (Default: ``128``)
    // 
    //     Examples
    //         >>> resblock = ResBlock()
    //         >>> input = torch.rand(10, 128, 512)  # a random spectrogram
    //         >>> output = resblock(input)  # shape: (10, 128, 512)
    //     
    public class ResBlock : nn.Module
    {
        public nn.Module resblock_model;

        public ResBlock(string name, int n_freq = 128) : base(name)
        {
            this.resblock_model = nn.Sequential(
                nn.Conv1d(inputChannel: n_freq, outputChannel: n_freq, kernelSize: 1, bias: false),
                nn.BatchNorm1d(n_freq),
                nn.ReLU(inPlace: true),
                nn.Conv1d(inputChannel: n_freq, outputChannel: n_freq, kernelSize: 1, bias: false),
                nn.BatchNorm1d(n_freq));
            RegisterComponents();
        }

        // Pass the input through the ResBlock layer.
        //         Args:
        //             specgram (Tensor): the input sequence to the ResBlock layer (n_batch, n_freq, n_time).
        // 
        //         Return:
        //             Tensor shape: (n_batch, n_freq, n_time)
        //         
        public override Tensor forward(Tensor specgram)
        {
            return this.resblock_model.forward(specgram) + specgram;
        }
    }

    // MelResNet layer uses a stack of ResBlocks on spectrogram.
    // 
    //     Args:
    //         n_res_block: the number of ResBlock in stack. (Default: ``10``)
    //         n_freq: the number of bins in a spectrogram. (Default: ``128``)
    //         n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
    //         n_output: the number of output dimensions of melresnet. (Default: ``128``)
    //         kernelSize: the number of kernel size in the first Conv1d layer. (Default: ``5``)
    // 
    //     Examples
    //         >>> melresnet = MelResNet()
    //         >>> input = torch.rand(10, 128, 512)  # a random spectrogram
    //         >>> output = melresnet(input)  # shape: (10, 128, 508)
    //     
    public class MelResNet : nn.Module
    {
        public readonly nn.Module melresnet_model;

        public MelResNet(
            string name,
            int n_res_block = 10,
            int n_freq = 128,
            int n_hidden = 128,
            int n_output = 128,
            int kernel_size = 5) : base(name)
        {
            var modules = new List<nn.Module>();
            modules.Add(nn.Conv1d(inputChannel: n_freq, outputChannel: n_hidden, kernelSize: kernel_size, bias: false));
            modules.Add(nn.BatchNorm1d(n_hidden));
            modules.Add(nn.ReLU(inPlace: true));
            for (int i = 0; i < n_res_block; i++) {
                modules.Add(new ResBlock("resblock", n_hidden));
            }
            modules.Add(nn.Conv1d(inputChannel: n_hidden, outputChannel: n_output, kernelSize: 1));
            this.melresnet_model = nn.Sequential(modules);
            RegisterComponents();
        }

        // Pass the input through the MelResNet layer.
        //         Args:
        //             specgram (Tensor): the input sequence to the MelResNet layer (n_batch, n_freq, n_time).
        // 
        //         Return:
        //             Tensor shape: (n_batch, n_output, n_time - kernel_size + 1)
        //         
        public override Tensor forward(Tensor specgram)
        {
            return this.melresnet_model.forward(specgram);
        }
    }

    // Upscale the frequency and time dimensions of a spectrogram.
    // 
    //     Args:
    //         time_scale: the scale factor in time dimension
    //         freq_scale: the scale factor in frequency dimension
    // 
    //     Examples
    //         >>> stretch2d = Stretch2d(time_scale=10, freq_scale=5)
    // 
    //         >>> input = torch.rand(10, 100, 512)  # a random spectrogram
    //         >>> output = stretch2d(input)  # shape: (10, 500, 5120)
    //     
    public class Stretch2d : nn.Module
    {
        public int freq_scale;
        public int time_scale;

        public Stretch2d(string name, int time_scale, int freq_scale) : base(name)
        {
            this.freq_scale = freq_scale;
            this.time_scale = time_scale;
            this.RegisterComponents();
        }

        // Pass the input through the Stretch2d layer.
        // 
        //         Args:
        //             specgram (Tensor): the input sequence to the Stretch2d layer (..., n_freq, n_time).
        // 
        //         Return:
        //             Tensor shape: (..., n_freq * freq_scale, n_time * time_scale)
        //         
        public override Tensor forward(Tensor specgram)
        {
            //return specgram.repeat_interleave(this.freq_scale, -2).repeat_interleave(this.time_scale, -1);
            var output = repeat_interleave(specgram, this.freq_scale, -2);
            output = repeat_interleave(output, this.time_scale, -1);
            return output;
        }

        private static Tensor repeat_interleave(Tensor input, int repeats, int dim)
        {
            var output = input.unsqueeze(dim);
            var repeats_array = new long[output.dim()];
            for (int i = 0; i < repeats_array.Length; i++) repeats_array[i] = 1;
            repeats_array[repeats_array.Length + dim] = repeats;
            output = output.repeat(repeats_array);
            var shape = input.shape;
            shape[shape.Length + dim] *= repeats;
            return output.reshape(shape);
        }
    }

    // Upscale the dimensions of a spectrogram.
    // 
    //     Args:
    //         upsample_scales: the list of upsample scales.
    //         n_res_block: the number of ResBlock in stack. (Default: ``10``)
    //         n_freq: the number of bins in a spectrogram. (Default: ``128``)
    //         n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
    //         n_output: the number of output dimensions of melresnet. (Default: ``128``)
    //         kernelSize: the number of kernel size in the first Conv1d layer. (Default: ``5``)
    // 
    //     Examples
    //         >>> upsamplenetwork = UpsampleNetwork(upsample_scales=[4, 4, 16])
    //         >>> input = torch.rand(10, 128, 10)  # a random spectrogram
    //         >>> output = upsamplenetwork(input)  # shape: (10, 128, 1536), (10, 128, 1536)
    //     
    public class UpsampleNetwork
        : nn.Module
    {
        public readonly int indent;
        public readonly MelResNet resnet;
        public readonly Stretch2d resnet_stretch;
        public readonly int total_scale;
        public readonly nn.Module upsample_layers;

        public UpsampleNetwork(
            string name,
            int[] upsample_scales,
            int n_res_block = 10,
            int n_freq = 128,
            int n_hidden = 128,
            int n_output = 128,
            int kernel_size = 5) : base(name)
        {
            var total_scale = 1;
            foreach (var upsample_scale in upsample_scales) {
                total_scale *= upsample_scale;
            }
            this.total_scale = total_scale;
            this.indent = (kernel_size - 1) / 2 * total_scale;
            this.resnet = new MelResNet("melresnet", n_res_block, n_freq, n_hidden, n_output, kernel_size);
            this.resnet_stretch = new Stretch2d("stretch2d", total_scale, 1);
            var up_layers = new List<nn.Module>();
            foreach (var scale in upsample_scales) {
                var stretch = new Stretch2d("stretch2d", scale, 1);
                var conv = nn.Conv2d(inputChannel: 1, outputChannel: 1, kernelSize: (1, scale * 2 + 1), padding: (0, scale), bias: false);
                torch.nn.init.constant_(conv.weight, 1.0 / (scale * 2 + 1));
                up_layers.Add(stretch);
                up_layers.Add(conv);
            }
            this.upsample_layers = nn.Sequential(up_layers);
            this.RegisterComponents();
        }

        // Pass the input through the UpsampleNetwork layer.
        // 
        //         Args:
        //             specgram (Tensor): the input sequence to the UpsampleNetwork layer (n_batch, n_freq, n_time)
        // 
        //         Return:
        //             Tensor shape: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale),
        //                           (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
        //         where total_scale is the product of all elements in upsample_scales.
        //         
        public new (Tensor, Tensor) forward(Tensor specgram)
        {
            var resnet_output = this.resnet.forward(specgram).unsqueeze(1);
            resnet_output = this.resnet_stretch.forward(resnet_output);
            resnet_output = resnet_output.squeeze(1);
            specgram = specgram.unsqueeze(1);
            var upsampling_output = this.upsample_layers.forward(specgram);
            upsampling_output = upsampling_output.squeeze(1)[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(this.indent, -this.indent)];
            return (upsampling_output, resnet_output);
        }
    }

    // WaveRNN model based on the implementation from `fatchord <https://github.com/fatchord/WaveRNN>`_.
    // 
    //     The original implementation was introduced in *Efficient Neural Audio Synthesis*
    //     [:footcite:`kalchbrenner2018efficient`]. The input channels of waveform and spectrogram have to be 1.
    //     The product of `upsample_scales` must equal `hop_length`.
    // 
    //     Args:
    //         upsample_scales: the list of upsample scales.
    //         n_classes: the number of output classes.
    //         hop_length: the number of samples between the starts of consecutive frames.
    //         n_res_block: the number of ResBlock in stack. (Default: ``10``)
    //         n_rnn: the dimension of RNN layer. (Default: ``512``)
    //         n_fc: the dimension of fully connected layer. (Default: ``512``)
    //         kernelSize: the number of kernel size in the first Conv1d layer. (Default: ``5``)
    //         n_freq: the number of bins in a spectrogram. (Default: ``128``)
    //         n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
    //         n_output: the number of output dimensions of melresnet. (Default: ``128``)
    // 
    //     Example
    //         >>> wavernn = WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
    //         >>> waveform, sample_rate = torchaudio.load(file)
    //         >>> # waveform shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
    //         >>> specgram = MelSpectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
    //         >>> output = wavernn(waveform, specgram)
    //         >>> # output shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
    //     
    public class WaveRNN : nn.Module
    {
        public int _pad;
        public nn.Module fc;
        public nn.Module fc1;
        public nn.Module fc2;
        public nn.Module fc3;
        public int hop_length;
        public int kernel_size;
        public int n_aux;
        public int n_bits;
        public int n_classes;
        public int n_rnn;
        public nn.Module relu1;
        public nn.Module relu2;
        public GRU rnn1;
        public GRU rnn2;
        public UpsampleNetwork upsample;

        public WaveRNN(
            string name,
            int[] upsample_scales,
            int n_classes,
            int hop_length,
            int n_res_block = 10,
            int n_rnn = 512,
            int n_fc = 512,
            int kernel_size = 5,
            int n_freq = 128,
            int n_hidden = 128,
            int n_output = 128) : base(name)
        {
            this.kernel_size = kernel_size;
            this._pad = (kernel_size % 2) == 1 ? kernel_size - 1 : kernel_size / 2;
            this.n_rnn = n_rnn;
            this.n_aux = n_output / 4;
            this.hop_length = hop_length;
            this.n_classes = n_classes;
            this.n_bits = (int)(Math.Log(this.n_classes) / Math.Log(2));
            var total_scale = 1;
            foreach (var upsample_scale in upsample_scales) {
                total_scale *= upsample_scale;
            }
            if (total_scale != this.hop_length) {
                throw new ArgumentException($"Expected: total_scale == hop_length, but found {total_scale} != {hop_length}");
            }
            this.upsample = new UpsampleNetwork("upsamplenetwork", upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size);
            this.fc = nn.Linear(n_freq + this.n_aux + 1, n_rnn);
            this.rnn1 = nn.GRU(n_rnn, n_rnn, batchFirst: true);
            this.rnn2 = nn.GRU(n_rnn + this.n_aux, n_rnn, batchFirst: true);
            this.relu1 = nn.ReLU(inPlace: true);
            this.relu2 = nn.ReLU(inPlace: true);
            this.fc1 = nn.Linear(n_rnn + this.n_aux, n_fc);
            this.fc2 = nn.Linear(n_fc + this.n_aux, n_fc);
            this.fc3 = nn.Linear(n_fc, this.n_classes);
            this.RegisterComponents();
        }

        // Pass the input through the WaveRNN model.
        // 
        //         Args:
        //             waveform: the input waveform to the WaveRNN layer (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
        //             specgram: the input spectrogram to the WaveRNN layer (n_batch, 1, n_freq, n_time)
        // 
        //         Return:
        //             Tensor: shape (n_batch, 1, (n_time - kernel_size + 1) * hop_length, n_classes)
        //         
        public override Tensor forward(Tensor waveform, Tensor specgram)
        {
            if (waveform.size(1) != 1) {
                throw new ArgumentException("Require the input channel of waveform is 1");
            }
            if (specgram.size(1) != 1) {
                throw new ArgumentException("Require the input channel of specgram is 1");
            }
            // remove channel dimension until the end
            waveform = waveform.squeeze(1);
            specgram = specgram.squeeze(1);
            var batch_size = waveform.size(0);
            var h1 = torch.zeros(1, batch_size, this.n_rnn, dtype: waveform.dtype, device: waveform.device);
            var h2 = torch.zeros(1, batch_size, this.n_rnn, dtype: waveform.dtype, device: waveform.device);
            // output of upsample:
            // specgram: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale)
            // aux: (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
            Tensor aux;
            (specgram, aux) = this.upsample.forward(specgram);
            specgram = specgram.transpose(1, 2);
            aux = aux.transpose(1, 2);
            var aux_idx = (from i in Enumerable.Range(0, 5)
                           select (this.n_aux * i)).ToList();
            var a1 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[0], aux_idx[1])];
            var a2 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[1], aux_idx[2])];
            var a3 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[2], aux_idx[3])];
            var a4 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[3], aux_idx[4])];
            var x = torch.cat(new Tensor[] {
                waveform.unsqueeze(-1),
                specgram,
                a1
            }, dimension: -1);
            x = this.fc.forward(x);
            var res = x;
            Tensor dummy;
            (x, dummy) = this.rnn1.forward(x, h1);
            x = x + res;
            res = x;
            x = torch.cat(new Tensor[] {
                x,
                a2
            }, dimension: -1);
            (x, dummy) = this.rnn2.forward(x, h2);
            x = x + res;
            x = torch.cat(new Tensor[] {
                x,
                a3
            }, dimension: -1);
            x = this.fc1.forward(x);
            x = this.relu1.forward(x);
            x = torch.cat(new Tensor[] {
                x,
                a4
            }, dimension: -1);
            x = this.fc2.forward(x);
            x = this.relu2.forward(x);
            x = this.fc3.forward(x);
            // bring back channel dimension
            return x.unsqueeze(1);
        }

        // Inference method of WaveRNN.
        // 
        //         This function currently only supports multinomial sampling, which assumes the
        //         network is trained on cross entropy loss.
        // 
        //         Args:
        //             specgram (Tensor):
        //                 Batch of spectrograms. Shape: `(n_batch, n_freq, n_time)`.
        //             lengths (Tensor or None, optional):
        //                 Indicates the valid length of each audio in the batch.
        //                 Shape: `(batch, )`.
        //                 When the ``specgram`` contains spectrograms with different durations,
        //                 by providing ``lengths`` argument, the model will compute
        //                 the corresponding valid output lengths.
        //                 If ``None``, it is assumed that all the audio in ``waveforms``
        //                 have valid length. Default: ``None``.
        // 
        //         Returns:
        //             (Tensor, Optional[Tensor]):
        //             Tensor
        //                 The inferred waveform of size `(n_batch, 1, n_time)`.
        //                 1 stands for a single channel.
        //             Tensor or None
        //                 If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
        //                 is returned.
        //                 It indicates the valid length in time axis of the output Tensor.
        //         
        public virtual (Tensor, Tensor?) infer(Tensor specgram, Tensor? lengths = null)
        {
            var device = specgram.device;
            var dtype = specgram.dtype;
            specgram = torch.nn.functional.pad(specgram, (this._pad, this._pad));
            Tensor aux;
            (specgram, aux) = this.upsample.forward(specgram);
            if (lengths is not null) {
                lengths = lengths * this.upsample.total_scale;
            }
            var output = new List<Tensor>();
            long b_size = specgram.size()[0];
            long seq_len = specgram.size()[2];
            var h1 = torch.zeros(new long[] { 1, b_size, this.n_rnn }, device: device, dtype: dtype);
            var h2 = torch.zeros(new long[] { 1, b_size, this.n_rnn }, device: device, dtype: dtype);
            var x = torch.zeros(new long[] { b_size, 1 }, device: device, dtype: dtype);
            var aux_split = new Tensor[4];
            for (int i = 0; i < 4; i++) {
                aux_split[i] = aux[TensorIndex.Colon, TensorIndex.Slice(this.n_aux * i, this.n_aux * (i + 1)), TensorIndex.Colon];
            }
            for (int i = 0; i < seq_len; i++) {
                var m_t = specgram[TensorIndex.Colon, TensorIndex.Colon, i];
                var a1_t = aux_split[0][TensorIndex.Colon, TensorIndex.Colon, 0];
                var a2_t = aux_split[0][TensorIndex.Colon, TensorIndex.Colon, 1];
                var a3_t = aux_split[0][TensorIndex.Colon, TensorIndex.Colon, 2];
                var a4_t = aux_split[0][TensorIndex.Colon, TensorIndex.Colon, 3];
                x = torch.cat(new Tensor[] {
                    x,
                    m_t,
                    a1_t
                }, dimension: 1);
                x = this.fc.forward(x);
                (_, h1) = this.rnn1.forward(x.unsqueeze(1), h1);
                x = x + h1[0];
                var inp = torch.cat(new Tensor[] {
                    x,
                    a2_t
                }, dimension: 1);
                (_, h2) = this.rnn2.forward(inp.unsqueeze(1), h2);
                x = x + h2[0];
                x = torch.cat(new Tensor[] {
                    x,
                    a3_t
                }, dimension: 1);
                x = F.relu(this.fc1.forward(x));
                x = torch.cat(new Tensor[] {
                    x,
                    a4_t
                }, dimension: 1);
                x = F.relu(this.fc2.forward(x));
                var logits = this.fc3.forward(x);
                var posterior = F.softmax(logits, dim: 1);
                x = torch.multinomial(posterior, 1).@float();
                // Transform label [0, 2 ** n_bits - 1] to waveform [-1, 1]
                x = 2 * x / (Math.Pow(2, this.n_bits) - 1.0) - 1.0;
                output.Add(x);
            }
            return (torch.stack(output).permute(1, 2, 0), lengths);
        }
    }
}
