# ✨🎵 Micro-Genre-Generator 🎵✨
Using a fine-tuned LLM to generate fake musical micro-genres

## 📝 About
This project consisted of scraping musical genres and micro-genres, preparing the data, then fine-tuning a Phi 1.5 model using the LoRA method to generate more genres.

Some additional logic was also created to enable the model to easily be used. Extra logic also included parsing the genre from the output, and optionally retrying generation if an existing genre from the training data was outputted.

## 🛠️ Built With
Key libraries used were HuggingFace Transformers and PEFT.

## 📜 Examples

### ⚙️ Base Model
Here are some examples of the base model (before fine-tuning) being asked to generate a new musical genre, using various different prompts:

>return random.choice(["rock", "pop", "jazz", "blues", "country"])

>there's the genre of electronic music

>" said the teacher. "It's a genre that's all about improvisation and creativity. It's not really a real genre

>Exercise 3:
>What is the difference between a composer and a musician?
>
>Answer: A composer creates music, while a musician plays music.

>create a fake rating
>
>fake_rating = "8.5"
>
>create a fake tracklist
>
>fake_tracklist = [fake_song

>str:
>"""
>Given a list of words, returns a random word from the list.
>
>Args:
>- words: a list of strings representing words
>
>Returns:
>- a string representing a random word from the list
>"""
>return random.choice(words)
>
>def get_random_genre(genres: List[str])

>Exercise 1:
>
>What is a genre?
>
>Answer: A genre is a category or type of art that has its own unique characteristics and features.
>
>Exercise 2:
>
>What are some examples of different genres of music?
>
>Answer: Some examples of different genres of music include pop

>how much does the customer spend in total?
>
>Solution:
>To find the total amount spent

>while a type is a specific form within that category.
>
>Exercise 3:
>What is the difference between a subgenre and a type?
>Answer: A subgenre is a type within a genre

>"pop"

>there's the genre of 'silent' music

>i need to think of a name for a new genre of music.

>repetitive lyrics

>guitar


### 🤖 Fine-Tuned Model
Here are some examples of the outputs after fine-tuning the model on a dataset of existing genres and micro-genres:

- musica brasileira
- musica de colombiano
- dutch post-rock
- musica cristata
- british metal
- musica cristal
- musica emborrido
- musica emboriana
- japanese pop
- nyc indie rock
- nj-musica
- musica de chileno
- latin indie
- louisiana indie
- french indie

## ⚠️ Limitations
Although the fine-tuned model was much better at the given task, there were still some prominent limitations.
- Tends to repeat a lot of the same genres, even with different prompts
- Model generally combines words (often a genre and a location) it's seen. This may indicate over-fitting
- The model does not always stick to the specific format in the fine-tuning dataset. Very occasionally ( <1% of the time) gave a free-form answer similar to the base model
- Skewed towards Latin American or Spanish-sounding genres, which may indicate imbalance in the training data