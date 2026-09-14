GeoBench is a benchmark for evaluating how well large language models can geolocate images, through the context of GeoGuessr. This project tests whether models can generalize beyond their primary training modalities to perform spatial reasoning tasks.

# **[Leaderboard](https://geobench.org)**

![](img/leaderboard.png)
For an in-depth explanation of the results, covering things like model behavior and reasoning, see my [writeup](https://ccmdi.com/blog/GeoBench).

# Installation
```
git clone https://github.com/ccmdi/geobench.git
cd geobench
pip install -r requirements.txt
```

Setup your `.env` based on `SAMPLE.env` for whichever model providers you wish to test for (e.g. `ANTHROPIC_API_KEY` must be set to test Claude). Instructions for setting up `NCFA` can be found [here](https://github.com/EvickaStudio/GeoGuessr-API?tab=readme-ov-file#authentication).

## Create a dataset
```
python dataset.py --num <n> --output <test name> --map <geoguessr map id>
```

## Test a model
```
python geobench.py --dataset <test name> --model <model name>
```

Models go by their class name in `models.py`. Claude 3.5 Haiku goes by `Claude3_5Haiku`, for instance.

## Test yourself
```
python human/quiz.py --dataset <test name> --num 10
```

Ten of the dataset's locations in the browser: drop a pin, write your reasoning,
and see the answer together with what every model said about the same image. It
saves the same run format the models do, so you land in the comparisons beside
them. See [`human/README.md`](human/README.md).

## Compare guesses
Running the `browser/main.py` script and opening `visualization.html` can show you all guesses for a location made by the models.

![](img/visualization.png)
