# Mining-Pokemon-GO-through-Twitter

Twitter is an easy way to get a large and rich dataset in a short period of time. One can obtain data through one of Twitter streaming APIs. For this project I decided to use the keyword "Pokemon", since Pokemon Go was very popular game at the time when I started this project. I managed to collect around 200,000 statuses containing "Pokemon" in one day. I used several tools to collect and analyse data.

TOOLS

 * Data collected through Twitter Streaming API Tweepy and stored in a SQLite database.

 * Data analysis done using SQLite Studio and Pandas.

 * Visualization done using Matplotplib and Plotly.

 * Natural Language Processing done with TextBlob.

It appears that English was the most popular language used in the statuses containing "Pokemon". Also IPhone and Android were leading devices used to send statuses from. Probably, people who played "Pokemon" on their mobile phones were more keen to share their excitement than people used other services. It is interesting that IPhone was more popular device than Android despite the fact that there are much more Android devices on the gobal market. This is probably due to the fact that most statuses came from the USA, where IPhone is more popular than Android.

Popularity of Pokemon GO by

 * language:

![Language][lan]
[lan]: https://github.com/ofialko/Data-Science-Projects/blob/master/Minning_PokemonGO/Figures/Language.png


 * source

![Language][sou]
[sou]: https://github.com/ofialko/Data-Science-Projects/blob/master/Minning_PokemonGO/Figures/Sources.png

 * location

![Language][loc]
[loc]: https://github.com/ofialko/Data-Science-Projects/blob/master/Minning_PokemonGO/Figures/Location.png

Pokemon was very popular game at the time I was doing this project. This is well supported by sentimental analysis of the statuses written in English. As can be seen from the pie-chart below there were much more positive statuses than negative.

![Language][sen]
[sen]: https://github.com/ofialko/Data-Science-Projects/blob/master/Minning_PokemonGO/Figures/Sentiment.png
