package neuronalnetworkapi.layer;

import java.io.IOException;
import java.io.PrintWriter;

public class ConvolutionalLayer extends Layer3D{
    
    double[][][] filter;

    public ConvolutionalLayer(int inputDepth, int inputLen, int stride, int padding, int nrFilters, int filterSize) {
        super(inputDepth, inputLen, filterSize, stride, padding, nrFilters);
        activation = (in, out) -> relu(in, out);
        dActivation = (in, out) -> dRelu(in, out);
        filter = new double[nrFilters][filterSize][filterSize];
        for(int i = 0; i < nrFilters; i++) {
            for(int j = 0; j < filterSize; j++) {
                for(int k = 0; k < filterSize; k++) {
                    // init randomly with number between -0.01 and +0.01
                    filter[i][j][k] = (Math.random() - 0.5) * 0.02;
                }
            }
        }
    }
    
    @Override
    public double[] compute(double[] input) {
        for(int i = 0; i < outputDepth; i++) {
            for(int j = 0; j < inputDepth; j++) {
                for(int k = 0; k < outputLen; k++) {
                    for(int l = 0; l < outputLen; l++) {
                        double val = 0.0;
                        for(int m = 0; m < kernelSize; m++) {
                            for(int n = 0; n < kernelSize; n++) {
                                val += input[j * inputLen * inputLen + (k + m) * inputLen + l + n] * filter[i][m][n];
                            }
                        }
                        z[i * outputLen * outputLen + k * outputLen + l] = val;
                    }
                }
            }
        }
        activation.accept(z, a);
        return a;
    }

    @Override
    public void adaptWeights() {
        int outputLenSquare = outputLen * outputLen;
        for(int i = 0; i < outputDepth; i++) {
            for(int j = 0; j < kernelSize; j++) {
                for(int k = 0; k < kernelSize; k++) {
                    for(int l = 0; l < outputLenSquare; l++) {
                        filter[i][j][k] -= learningRate * delta[i * outputLenSquare + l];
                    }
                }
            }
        }
    }

    @Override
    protected void backPropagatePrevLayer(double[] dz, double[] delta) {
        for(int i = 0; i < inputDepth; i++) {
            for(int j = 0; j < outputDepth; j++) {
                double[][] convDelta = fullConvolve(deltaTo2D(this.delta, i), j);
                for(int m = 0; m < inputLen; m++) {
                    for(int n = 0; n < inputLen; n++) {
                        delta[i * inputLen * inputLen + m * inputLen + n] += convDelta[m][n];
                    }
                }
            }
        }
    }

    @Override
    public void print(PrintWriter writer) throws IOException{
        for(double[][] filt : filter) {
            for(double[] row : filt) {
                for(double val : row) {
                    writer.print(val);
                    writer.print("   ");
                }
                writer.println();
            }
            writer.println();
        }
        writer.flush();
    }
    
    private double[][] deltaTo2D(double[] delta, int index) {
        double[][] result = new double[outputLen][outputLen];
        for(int i = 0; i < outputLen; i++) {
            System.arraycopy(delta, index  * outputLen * outputLen + i * outputLen, result[i], 0, outputLen);
        }
        return result;
    }
    
    private double[][] fullConvolve(double[][] input, int f) {
        double[][] res = new double[inputLen][inputLen];
        double[][] paddedInput = pad(input, kernelSize - 1, kernelSize - 1);
        double[][] flippedFilter = flip(filter[f]);
        for(int i = 0; i < inputLen; i++) {
            for(int j = 0; j < inputLen; j++) {
                double val = 0.0;
                for(int m = 0; m < kernelSize; m++) {
                    for(int n = 0; n < kernelSize; n++) {
                        val += paddedInput[i + m][j + n] * flippedFilter[m][n];
                    }
                }
                res[i][j] = val;
            }
        }
        return res;
    }
    
    private double[][] flip(double[][] input) {
        double[][] output = new double[input[0].length][input.length];
        for(int i = 0; i < output.length; i++) {
            for(int j = 0; j < output[i].length; j++) {
                output[i][j] = input[output[i].length - i - 1][output.length - j - 1];
            }
        }
        return output;
    }
    
    private double[][] pad(double[][] input, int horizontal, int vertical) {
        double[][] output = new double[input.length + 2 * vertical][input[0].length + 2 * horizontal];
        for(int i = 0; i < input.length; i++) {
            System.arraycopy(input[i], 0, output[i + vertical], horizontal, input[i].length);
        }
        return output;
    }
    
    private void relu(double[] in, double[] out) {
        for(int i = 0; i < out.length; i++) {
            out[i] = Math.max(0, in[i]);
        }
    }
    
    private void dRelu(double[] in, double[] out) {
        for(int i = 0; i < in.length; i++) {
            if(in[i] > 0) {
                out[i] = 1;
            } else {
                out[i] = 0;
            }
        }
    }
}
