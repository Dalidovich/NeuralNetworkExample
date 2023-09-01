using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NeuralNetworkExample
{
    public class NeuralNetwork
    {
        public double[][,] layers { get; set; }
        public double[][,] weights { get; set; }
        public double smoothing { get; set; }
        public decimal errorTolerance { get; set; }
        private Random rnd = new Random();

        public NeuralNetwork(int countInputNeirons, int[] hidenLayersArray, int countOutputNeirons, double smoothing, decimal errorTolerance)
        {
            this.smoothing = smoothing;
            this.errorTolerance = errorTolerance;
            layers = new double[hidenLayersArray.Length + 2][,];
            weights = new double[hidenLayersArray.Length + 1][,];
            createLayers(countInputNeirons, hidenLayersArray, countOutputNeirons);
            createWeights();
        }

        public NeuralNetwork()
        {
        }

        private void createWeights()
        {
            for (int i = 0; i < layers.Length - 2; i++)
            {
                weights[i] = new double[layers[i].GetLength(0), layers[i + 1].GetLength(0) - 1];
            }
            weights[weights.Length - 1] = new double[layers[layers.Length - 2].GetLength(0), layers[layers.Length - 1].GetLength(0)];

            for (int i = 0; i < weights.Length; i++)
            {
                for (int k = 0; k < weights[i].GetLength(0) - 1; k++)
                {
                    for (int z = 0; z < weights[i].GetLength(1); z++)
                    {
                        weights[i][k, z] = rnd.Next(0, 2) - rnd.NextDouble();
                    }
                }

                for (int z = 0; z < weights[i].GetLength(1); z++)
                {
                    weights[i][weights[i].GetLength(0) - 1, z] = 1;
                }
            }

        }

        private void createLayers(int countInputNeirons, int[] hidenLayersArray, int countOutputNeirons)
        {
            layers[0] = new double[countInputNeirons + 1, 2];
            layers[0][countInputNeirons, 0] = 1;
            for (int i = 0; i < hidenLayersArray.Length; i++)
            {
                layers[i + 1] = new double[hidenLayersArray[i] + 1, 2];
                layers[i + 1][layers[i + 1].GetLength(0) - 1, 0] = 1;
            }
            layers[layers.Length - 1] = new double[countOutputNeirons, 2];
        }

        public void learn(double[,] expected, double[,] inputs, int saveEpoch = 0)
        {
            var error = 90.0m;
            ulong iter = 0;
            do
            {
                for (int i = 0; i < inputs.GetLength(0); i++)
                {
                    double[] inputsSignals = new double[inputs.GetLength(1)];
                    for (int k = 0; k < inputs.GetLength(1); k++)
                    {
                        inputsSignals[k] = inputs[i, k];
                    }
                    double[] expectedSignals = new double[expected.GetLength(1)];
                    for (int k = 0; k < expected.GetLength(1); k++)
                    {
                        expectedSignals[k] = expected[i, k];
                    }
                    inputSignalInlayer(layers[0], inputsSignals);
                    feedForwards();
                    error = findError(layers.Last(), expectedSignals);
                    findErrors();
                    fixWeight();
                }
                if (saveEpoch != 0 && iter % (ulong)saveEpoch == 0)
                {
                    saveNN("NN");
                }
                if (iter % 1000 == 0)
                    Console.WriteLine($"iteration: {iter},\t error - {error}\t");
                iter++;
            }
            while (error > Convert.ToDecimal(errorTolerance) || error < Convert.ToDecimal(-errorTolerance));
        }

        public void inputSignalInlayer(double[,] layer, params double[] signals)
        {
            for (int i = 0; i < layer.GetLength(0) - 1; i++)
            {
                layer[i, 0] = signals[i];
            }
        }

        public void feedForwards()
        {
            for (int i = 0; i < layers.Length - 1; i++)
            {
                forWards(layers[i], weights[i], layers[i + 1]);
            }
        }

        public decimal findError(double[,] layer, double[] expected)
        {
            double[] differens = new double[layer.GetLength(0)];
            if (layer.GetLength(0) != 1)
            {
                for (int i = 0; i < layer.GetLength(0); i++)
                {
                    layer[i, 1] = expected[i] - layer[i, 0];
                    differens[i] = layer[i, 1] * layer[i, 1];
                }
            }
            else
            {
                layer[0, 1] = expected[0] - layer[0, 0];
                differens[0] = layer[0, 1];
            }

            return Convert.ToDecimal(differens.Sum());
        }

        public void findErrors()
        {
            for (int z = layers.Length - 2; z >= 0; z--)
            {
                for (int i = 0; i < weights[z].GetLength(0) - 1; i++)
                {
                    layers[z][i, 1] = 0;
                    for (int j = 0; j < weights[z].GetLength(1); j++)
                    {
                        layers[z][i, 1] = layers[z][i, 1] + weights[z][i, j] * layers[z + 1][j, 1];
                    }
                }
            }
        }

        public void fixWeight()
        {
            for (int z = layers.Length - 2; z >= 0; z--)
            {
                for (int i = 0; i < weights[z].GetLength(1); i++)
                {
                    for (int j = 0; j < weights[z].GetLength(0) - 1; j++)
                    {
                        weights[z][j, i] = weights[z][j, i] + smoothing * layers[z + 1][i, 1] * layers[z][j, 0] * SigmoidDx(layers[z + 1][i, 0]);
                    }
                }
            }
        }

        public void forWards(double[,] startLayer, double[,] weights, double[,] endLayer)
        {
            int countInputNeuron = weights.GetLength(0);
            int countOutputNeuron = weights.GetLength(1);

            for (int i = 0; i < countOutputNeuron; i++)
            {
                endLayer[i, 0] = 0;
                for (int j = 0; j < countInputNeuron; j++)
                {
                    endLayer[i, 0] = endLayer[i, 0] + startLayer[j, 0] * weights[j, i];
                }
                //пропускаем через функцию активации
                endLayer[i, 0] = Sigmoid(endLayer[i, 0]);
            }
        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid * (1 - sigmoid);
            return result;
        }

        public void saveNN(string fileName)
        {
            if (fileName == "") return;
            string json = JsonConvert.SerializeObject(this);
            File.WriteAllText($"{fileName}.json", json);
        }

        public static NeuralNetwork loadNN(string filename)
        {
            var nn = new NeuralNetwork();
            string jsonAll = File.ReadAllText($"{filename}.json");
            dynamic staff = JObject.Parse(jsonAll);
            nn = JsonConvert.DeserializeObject<NeuralNetwork>(staff.ToString());

            return nn;
        }
    }
}
