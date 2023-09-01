namespace NeuralNetworkExample.ExampleImplement.DataSection.Implement
{
    public class DataNNSectionWithSymbols
    {
        public bool GREEN = false;
        public string GREEN_FILL = ".";
        public string GREEN_POINT = "G";
        public string BlUE_FILL = "@";
        public string BlUE_POINT = "_";
        public string CURSOR = "C";

        public readonly int _width;
        public readonly int _height;

        private int _cursorX = 0;
        private int _cursorY = 0;

        public List<Point> Points { get; set; }

        public DataNNSectionWithSymbols(int width, int height)
        {
            _width = width;
            _height = height;

            Points = new List<Point>();
        }

        public void Visualize()
        {
            Console.SetWindowSize(_width, _height);
            Console.SetBufferSize(_width, _height);
            Console.CursorVisible = false;

            var nn = new NeuralNetwork(2, new[] { 5 }, 1, 0.1, 0.1m);

            while (true)
            {
                CheckControls();

                for (int i = 0; i < Points.Count; i++)
                {
                    var point = Points[i];
                    nn.inputSignalInlayer(nn.layers.First(), (float)point.X / _width, (float)point.Y / _height);
                    nn.feedForwards();

                    var expect = new double[] { point.flag ? 1 : 0 };

                    nn.findError(nn.layers.Last(), expect);
                    nn.findErrors();
                    nn.fixWeight();
                }

                string text = "";

                for (int i = 0; i < _height; i++)
                {
                    for (int k = 0; k < _width; k++)
                    {
                        if (_cursorX == k && _cursorY == i)
                        {
                            text += CURSOR;
                        }
                        else if (Points.Count(x => x.X == k && x.Y == i) != 0)
                        {
                            if (Points.Where(x => x.X == k && x.Y == i).First().flag)
                            {
                                text += BlUE_POINT;
                            }
                            else
                            {
                                text += GREEN_POINT;
                            }
                        }
                        else
                        {
                            nn.inputSignalInlayer(nn.layers.First(), (float)k / _width, (float)i / _height);
                            nn.feedForwards();

                            text += nn.layers.Last()[0, 0] > 0.5 ? BlUE_FILL : GREEN_FILL;
                        }
                    }
                }

                Console.SetCursorPosition(0, 0);
                Console.Write(text);
            }
        }

        void CheckControls()
        {
            if (Console.KeyAvailable)
            {
                ConsoleKey consoleKey = Console.ReadKey(true).Key;
                switch (consoleKey)
                {
                    case ConsoleKey.W:
                        if (_cursorY != 0)
                            _cursorY -= 1;
                        break;
                    case ConsoleKey.A:
                        if (_cursorX != 0)
                            _cursorX -= 1;
                        break;
                    case ConsoleKey.D:
                        if (_cursorX != _width - 1)
                            _cursorX += 1;
                        break;
                    case ConsoleKey.S:
                        if (_cursorY != _height - 1)
                            _cursorY += 1;
                        break;
                    case ConsoleKey.B:
                        Points.Add(new Point(_cursorX, _cursorY, !GREEN));
                        break;
                    case ConsoleKey.G:
                        Points.Add(new Point(_cursorX, _cursorY, GREEN));
                        break;
                    default:
                        break;
                }
            }
        }
    }
}
