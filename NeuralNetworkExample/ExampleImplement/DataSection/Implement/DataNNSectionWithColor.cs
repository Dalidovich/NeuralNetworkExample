namespace NeuralNetworkExample.ExampleImplement.DataSection.Implement
{
    public class DataNNSectionWithColor
    {
        public bool GREEN = false;

        public readonly int _width;
        public readonly int _height;

        private int _cursorX = 0;
        private int _cursorY = 0;

        public char WriteSymbol = '@';

        public List<Point> Points { get; set; }

        public DataNNSectionWithColor(int width, int height)
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
                Console.SetCursorPosition(0, 0);

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

                for (int i = 0; i < _height; i++)
                {
                    for (int k = 0; k < _width; k++)
                    {
                        if (_cursorX == k && _cursorY == i)
                        {
                            Console.ForegroundColor = ConsoleColor.DarkRed;
                            Console.Write(WriteSymbol);
                            Console.ResetColor();
                        }
                        else if (Points.Count(x => x.X == k && x.Y == i) != 0)
                        {
                            if (Points.Where(x => x.X == k && x.Y == i).First().flag)
                            {
                                Console.ForegroundColor = ConsoleColor.DarkBlue;
                                Console.Write(WriteSymbol);
                                Console.ResetColor();
                            }
                            else
                            {
                                Console.ForegroundColor = ConsoleColor.DarkGreen;
                                Console.Write(WriteSymbol);
                                Console.ResetColor();
                            }
                        }
                        else
                        {
                            nn.inputSignalInlayer(nn.layers.First(), (float)k / _width, (float)i / _height);
                            nn.feedForwards();

                            Console.ForegroundColor = nn.layers.Last()[0, 0] > 0.5 ? ConsoleColor.Blue : ConsoleColor.Green;
                            Console.Write(WriteSymbol);
                            Console.ResetColor();
                        }
                    }
                }
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
