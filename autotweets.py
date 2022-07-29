from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import random
import tweepy
from tweepy import OAuthHandler
from dotenv import load_dotenv
import os
import time
import random, json
import urllib3, re

load_dotenv()

ALLOW_NEW_LINES = False
EPOCHS = 4

# <--- Enter your credentials (don't share with anyone) --->
HUB_TOKEN = os.environ["HUB_TOKEN"]
consumer_key = os.environ["CONSUMER_KEY"]
consumer_secret = os.environ["CONSUMER_SECRET"]


def fix_text(text):
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    return text


def clean_tweet(tweet, allow_new_lines=ALLOW_NEW_LINES):
    bad_start = ["http:", "https:"]
    for w in bad_start:
        tweet = re.sub(f" {w}\\S+", "", tweet)  # removes white space before url
        tweet = re.sub(f"{w}\\S+ ", "", tweet)  # in case a tweet starts with a url
        tweet = re.sub(f"\n{w}\\S+ ", "", tweet)  # in case the url is on a new line
        tweet = re.sub(
            f"\n{w}\\S+", "", tweet
        )  # in case the url is alone on a new line
        tweet = re.sub(f"{w}\\S+", "", tweet)  # any other case?
    tweet = re.sub(" +", " ", tweet)  # replace multiple spaces with one space
    if not allow_new_lines:  # TODO: predictions seem better without new lines
        tweet = " ".join(tweet.split())
    return tweet.strip()


def boring_tweet(tweet):
    "Check if this is a boring tweet"
    boring_stuff = ["http", "@", "#"]
    not_boring_words = len(
        [None for w in tweet.split() if all(bs not in w.lower() for bs in boring_stuff)]
    )
    return not_boring_words < 3


# authenticate
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)


def create_dataset():
    # <--- Enter the screen name of the user you will download your dataset from --->
    handles = os.environ["HANDLES"].split(",")
    alltweets = []

    cool_tweets = []
    handles_processed = []
    raw_tweets = []
    user_names = []
    n_tweets_dl = []
    n_retweets = []
    n_short_tweets = []
    n_tweets_kept = []
    i = 0
    for handle in handles:
        if handle in handles_processed:
            continue
        i += 1
        handles_processed.append(handle)
        print(
            f"Downloading @{handle} tweets... This should take no more than a minute!"
        )
        http = urllib3.PoolManager(retries=urllib3.Retry(3))
        res = http.request(
            "GET",
            f"http://us-central1-huggingtweets.cloudfunctions.net/get_tweets?handle={handle}&force=1",
        )
        res = json.loads(res.data.decode("utf-8"))
        user_names.append(res["user_name"])

        all_tweets = res["tweets"]
        raw_tweets.append(all_tweets)
        curated_tweets = [fix_text(tweet) for tweet in all_tweets]
        # log_dl_tweets.clear_output(wait=True)
        print(f"{res['n_tweets']} tweets from @{handle} downloaded!\n\n")

        # create dataset
        clean_tweets = [clean_tweet(tweet) for tweet in curated_tweets]
        cool_tweets.append([tweet for tweet in clean_tweets if not boring_tweet(tweet)])

        # save count
        n_tweets_dl.append(str(res["n_tweets"]))
        n_retweets.append(str(res["n_RT"]))
        n_short_tweets.append(str(len(all_tweets) - len(cool_tweets[-1])))
        n_tweets_kept.append(str(len(cool_tweets[-1])))

        if len("<|endoftext|>".join(cool_tweets[-1])) < 6000:
            # need about 4000 chars for one data sample (but depends on spaces, etc)
            raise ValueError(
                f"Error: this user does not have enough tweets to train a Neural Network\n{res['n_tweets']} tweets downloaded, including {res['n_RT']} RT's and {len(all_tweets) - len(cool_tweets)} boring tweets... only {len(cool_tweets)} tweets kept!"
            )
        if len("<|endoftext|>".join(cool_tweets[-1])) < 40000:
            print(
                "<b>Warning: this user does not have many tweets which may impact the results of the Neural Network</b>\n"
            )

        print(
            f"{n_tweets_dl[-1]} tweets downloaded, including {n_retweets[-1]} RT's and {n_short_tweets[-1]} short tweets... keeping {n_tweets_kept[-1]} tweets"
        )

    print("Creating dataset...")
    # create a file based on multiple epochs with tweets mixed up
    seed_data = random.randint(0, 2 ** 32 - 1)
    dataRandom = random.Random(seed_data)
    total_text = "<|endoftext|>"
    all_handle_tweets = []
    epoch_len = max(len("".join(cool_tweet)) for cool_tweet in cool_tweets)
    for _ in range(EPOCHS):
        for cool_tweet in cool_tweets:
            dataRandom.shuffle(cool_tweet)
            current_tweet = cool_tweet
            current_len = len("".join(current_tweet))
            while current_len < epoch_len:
                for t in cool_tweet:
                    current_tweet.append(t)
                    current_len += len(t)
                    if current_len >= epoch_len:
                        break
            dataRandom.shuffle(current_tweet)
            all_handle_tweets.extend(current_tweet)
    total_text += "<|endoftext|>".join(all_handle_tweets) + "<|endoftext|>"

    with open(
        f"data_{'-'.join(sorted(handles_processed))}_train.txt", "w", encoding="utf-8"
    ) as f:
        f.write(total_text)

    print("Dataset created!")


handle = input("Enter 1-3 twitter handles (seperated by commas): ")
handles = handle.split(",")
handles_formatted = []
for i in handles:
    i = i.strip()
    if i.startswith("@"):
        i = i[1:]
    handles_formatted.append(i)

print("Checking if handles are valid...")
for i in handles_formatted:
    try:
        user = api.get_user(i)
        print("Handle is valid! ({})".format(i))
    except:
        print("Invalid handle: " + i)
        exit()

os.environ["HANDLES"] = ",".join(handles_formatted)
hfuser = os.environ["HF_USER"]


print("Checking if dataset already exists...")
model = "-".join(sorted(handles_formatted))
model = f"{hfuser}/" + model
HUB_TOKEN = os.environ["HUB_TOKEN"]

try:
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    print("Dataset already exists! Continuing...")
except:
    yn = input(
        "Dataset does not exist. Would you like to create & finetune it? (y/n): "
    )
    if yn == "y":
        n = 6
        for _ in range(0, 5):
            n -= 1
            print(f"{n}...")
            time.sleep(1)
        print("Creating...")

        create_dataset()

        handle = "-".join(sorted(handles_formatted))

        os.system(
            f"python run_clm.py \
            --model_name_or_path gpt2 \
            --train_file data_{'-'.join(sorted(handles_formatted))}_train.txt \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --do_train \
            --overwrite_output_dir \
            --prediction_loss_only \
            --logging_steps 5 \
            --save_steps 0 \
            --seed {random.randint(0,2**32-1)} \
            --learning_rate 1.372e-4 \
            --push_to_hub \
            --hub_token {HUB_TOKEN} \
            --output_dir ./{handle}"
        )

        print("Done!\nRestart the program to continue.")
        exit()

    elif yn == "n":
        print("Exiting...")
        exit()
    else:
        print("Invalid input. Exiting...")
        exit()

LEARNING_RATE = 1.372e-4
ALLOW_NEW_LINES = False
seed = random.randint(0, 2 ** 32 - 1)
handles_processed = handles
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
block_size = tokenizer.model_max_length

training_args = TrainingArguments(
    output_dir="./out",
    overwrite_output_dir=True,
    do_train=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    prediction_loss_only=True,
    logging_steps=5,
    save_steps=0,
    seed=seed,
    learning_rate=LEARNING_RATE,
)


trainer = Trainer(
    model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator
)

while True:

    start = input("Prompt: ")

    start_with_bos = "<|endoftext|>" + start
    encoded_prompt = trainer.tokenizer(
        start_with_bos, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    encoded_prompt = encoded_prompt.to(trainer.model.device)

    # prediction
    output_sequences = trainer.model.generate(
        input_ids=encoded_prompt,
        max_length=160,
        min_length=10,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=10,
    )
    generated_sequences = []
    predictions = []

    # decode prediction
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = trainer.tokenizer.decode(
            generated_sequence,
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )
        if not ALLOW_NEW_LINES:
            limit = text.find("\n")
            text = text[: limit if limit != -1 else None]
        generated_sequences.append(text.strip())

    for i, g in enumerate(generated_sequences):
        predictions.append([start, g])

    n = 0
    print("\n")
    for i in predictions:
        n += 1
        print("{}. {}".format(n, i))
    print("\n")

    while True:

        yn = input("\nWould you like to tweet any of these? (y/n): ")

        while True:
            if yn == "y":
                tweet_idx = input(
                    "Enter the number of the result you would like to tweet: "
                )
                try:
                    tweet = generated_sequences[int(tweet_idx) - 1]
                    api.update_status(tweet)
                    print("Tweeted: {}".format(tweet))
                    break
                except:
                    print("Invalid index or credentials. Try again.")
                    continue
        if yn == "n":
            print("\nNo tweet sent.")
            break
        else:
            print("\nInvalid input.")
            continue
