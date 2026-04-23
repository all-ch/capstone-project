import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity 
import matplotlib.pyplot as plt

conference_path = "data/data.xlsx"
df = pd.read_excel(conference_path, sheet_name="WCF Presenters")

speech_cols = [
    "Presentation_Text1",
    "Presentation_Text2",
    "Presentation_Text3",
    "Presentation_Text4",
    "Presentation_Text5",
]
# make sure missing columns do not crash the code
speech_cols = [col for col in speech_cols if col in df.columns]

# combine all text columns into one speech string per row
df["full_speech"] = (
    df[speech_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
)

# remove rows where the combined speech is empty
df = df[df["full_speech"] != ""].copy()

# https://sbert.net/ - "there is plenty of interesting analysis to do even with a pre-trained embeddings model"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ta suggested to use anchor words for both sides of the axis, secular vs religious
# push each anchor word through the model to get its embedding, and store in a list
# example usage: generate_anchor_embedding("prelim_religion_pos_anchors.csv", "prelim_religion_neg_anchors.csv")
def generate_anchor_embedding(pos_text, neg_text) -> list:
    pos_anchor_text = pd.read_csv(pos_text)["word"].tolist()
    neg_anchor_text = pd.read_csv(neg_text)["word"].tolist()
    return model.encode(pos_anchor_text), model.encode(neg_anchor_text)


def calculate_topic_vector(pos_embeddings, neg_embeddings) -> list:
    # calculate the average embedding for each side of the axis by taking the mean of the embeddings for the positive and negative anchor words
    pos_avg_embedding = pos_embeddings.mean(axis=0)
    neg_avg_embedding = neg_embeddings.mean(axis=0)
    # calculate the topic vector by subtracting the average negative embedding from the average positive embedding
    topic_vector = pos_avg_embedding - neg_avg_embedding
    return topic_vector


"""So far, I imagine that this work can be translated into your project as follows:
- The disease as discussed in time period T => the conference in year T
- Types of stigma => topics of interest that may be discussed at the conference

I thus propose that you come up with topics of interest yourself (in collaboration with your mentor),
 for instance "religion" (and others). You then would create a religion score for the conference over time,
   and use regression/smoothing splines/etc to understand the trend of religious discussion (and other topics) at this conference over time."""


def compute_speech_topic_score(document_text, topic_vector) -> float:
    # idea workflow(super basic, probably not what Allinn wants :P)
    # tokenize each word
    # get the embedding for each word
    # calculate the cosine similarity between each word embedding and pos and neg avg embeddings
    text_embedding = model.encode(document_text)
    topic_score = np.mean(cosine_similarity(text_embedding.reshape(1, -1), topic_vector.reshape(1, -1)))
    return topic_score


# TA suggested yearly scores for each topic
def compute_yearly_topic_scores(conference_data, topic_vector) -> dict:
    # guess of process
    # group the conference data by year
    # for each year, compute the average topic score for all documents in that year using the compute_speech_topic_score function
    # return a dictionary with years as keys and average topic scores as values

    yearly_topic_score = {}
    for year, group in conference_data.groupby("Conference_Year"):
        yearly_topic_score[year] = group["full_speech"].apply(lambda x: compute_speech_topic_score(x, topic_vector)).mean()
    return yearly_topic_score

def compute_word_level_topic_score(document_text, topic_vector) -> list:
    # tokenize each word
    words = document_text.split()
    # get the embedding for each word
    
    word_scores = []
    for word in words:
        word_embedding = model.encode(word)
        word_topic_score = cosine_similarity(word_embedding.reshape(1, -1), topic_vector.reshape(1, -1))[0][0]
        word_scores.append(word_topic_score)
    return word_scores



if __name__ == "__main__":
    neg_embeddings, pos_embeddings = generate_anchor_embedding(
        "data/prelim_religion_neg_anchors.csv", "data/prelim_religion_pos_anchors.csv"
    )
    religion_topic_vector = calculate_topic_vector(pos_embeddings, neg_embeddings)

    # EXAMPLE RUN: PLEASE IMPLEMENT INTO THE FUNCTIONS ABOVE :PRAY:
    # religious example speech topic score test by Dmitry Smirnov
    # hypothesis:
    '''biblical_text = """After I had given the purchase agreement to Baruch, son of Neriah, I prayed to the Lord: Oh, Lord God! You yourself made the heavens and earth by your great power and with your outstretched arm. Nothing is too difficult for you! 18 You show faithful love to thousands but lay the fathers’ iniquity on their sons’ laps after them, great and mighty God whose name is the Lord of Armies, the one great in counsel and powerful in action. Your eyes are on all the ways of the children of men a in order to reward each person according to his ways and as the result of his actions. You performed signs and wonders in the land of Egypt and still do today, both in Israel and among all mankind. You made a name for yourself, as is the case today. You brought your people Israel out of Egypt with signs and wonders, with a strong hand and an outstretched arm, and with great terror. You gave them this land you swore to give to their ancestors, a land flowing with milk and honey. They entered and possessed it, but they did not obey you or live according to your instructions. They failed to perform all you commanded them to do, and so you have brought all this disaster on them. Look! Siege ramps have come against the city to capture it, and the city, as a result of the sword, famine, and plague, has been handed over to the Chaldeans who are fighting against it. What you have spoken has happened. Look, you can see it!  Yet you, Lord God, have said to me, ‘Purchase the field and call in witnesses’ — even though the city has been handed over to the Chaldeans"""
    religious_anchors = """god god faith jesus christ god god"""
    non_religious_text =  """I express my appreciation to Dr. Jesus Hernandez, Dr. Allan Carlson, Dr. Fernando Milanes, Dr. Richard Wilkins, and to those of the organizing committee who extended this invitation and note their considerable contribution to the strength and well-being of families throughout the world. For those of you who don't know me, I would like to briefly introduce myself with some specifics. I grew up in what could be called a traditional family, blessed by the presence and commitment of my mother and father. My parents had six children, three boys and three girls. My father died when I was 20-years-old. A couple of years later, I married Debbie. We have now been married for 34 years. We have eight children, four girls and four boys. Six of our children are married. Our 19th grandchild was born a couple of weeks ago. I am a businessman. However, a few years ago I began devoting much of my time to certain societal matters of interest to me, principally the family. During this time, with a desire to better prepare myself to engage these societal issues, I enrolled at the Kennedy School of Government at Harvard University and there received a master's degree in public administration with a methodological area of concentration in leadership. Since that time, I have become affiliated with several excellent Non-Governmental Organizations that attempt to be an influence for good throughout the world, one of which is United Families International, a co-convening organization for this Congress as identified in your program. I hope you will excuse my sharing so much personal information. I do so because I must ask your further indulgence as I intend to take a risk and draw upon personal experiences relating to family, business, and education as I address you today on the subject of “the Contribution of Family Life to the Productivity of Economies and Companies.” Although economies and companies are sometimes influenced by circumstances beyond their control by things such as natural disasters, the presence or lack of certain natural resources, politics, acts of war, terrorism, etc., the productivity of economies and companies, whether local, regional, national or international, is directly related to the health, strength, and connectedness of the people who are members of the economic unit. Not surprisingly, these issues of health, strength, and connectedness are best nurtured and refined within families. Whether one is providing goods or services, the ability he or she possesses to engage others effectively and perform consistently will influence directly his or her level of production and corresponding contribution to economic growth. This could be called one's “productivity quotient,” or the degree to which one is able to produce more than one consumes. Let me try to say this another way. While much can be said about micro economics and macro economics, about demand curves and supply curves, about elasticity and inelasticity, or about equilibrium and dead weight losses, economic well-being is largely the product of individuals who are stable, net contributors to the world around them. An economic society or company composed entirely of capable, contributing members with high productivity quotients will produce much more than a society or company burdened with many members with low productivity quotients. Please do not misunderstand me. I do not limit the definition of a productivity quotient to monetary measurement. As a businessman, I know that for many, money is the only measure for keeping score. I disagree. Economic well-being cannot be defined merely in terms of the amount of money in one's purse. For example, productivity could include the ideas, social stability, or moral aptitude that one provides for the common good of the economic community. Under this definition, the value of a mother in teaching and nurturing another generation of well-balanced, contributing children may far exceed the value of the business icon's bank account. Public policy considerations, economic or otherwise, that ignore or diminish fundamental commitments to the perpetuation of the health and connectedness of the people are myopic and will ultimately damage society. With this understanding, it would make as much sense to attempt to describe economic productivity without acknowledging the components contributing to the long-term health and connectedness of the people as it would to attempt to describe a cake without identifying the ingredients or the recipe. A little less than a year ago, a group calling itself “The Commission on Children at Risk” presented an excellent work entitled, Hardwired to Connect, The New Scientific Case for Authoritative Communities. The group is composed of 33 children’s doctors, research scientists, and mental health and youth service professionals. The work was published jointly by the YMCA of the United States of America, Dartmouth Medical School, and the Institute for American Values. While its findings have profound implications for virtually every area of human existence, I would like to use them as a backdrop for my comments relating to the role of the family in contributing to economic growth. The report provides empirical evidence that humans are genetically and hormonally driven to connect to other people and to moral meaning. I emphasize that this is “not merely the result of social conditioning, but is instead an intrinsic aspect of the human experience.” The report suggests that this need is best met through what the commission calls “authoritative communities.” These are authoritative social institutions that include children and youth and treat them as ends in themselves, that are warm and nurturing, establish clear limits and expectations, that are multi-generational and have a long-term focus, that transmit a shared understanding of what it means to be a good person, that encourage spiritual and religious development, that teach love of neighbor, and are institutions where the core work is done by non-specialists. It is worth noting that the commission considers the family as “arguably the first and most basic association of civil society, and a centrally important example of what should be an authoritative community.” With that definition, the commission presents ten planks of the new scientific case for authoritative communities. I will not identify them all here, but want to mention five that I consider to be relevant to my topic. You won’t need to remember them specifically, but a general sense of what they convey will be helpful. 1. Nurturing or non-nurturing environments affect gene transcription and the development of brain circuitry. When children are held and loved, they become predisposed at cellular level to pass on good nurturing and physiological resilience to the next generation. In other words, generations can be affected by the nurturing that does or does not occur within a home. 2. Social contexts can alter genetic expression. Both “nature” and “nurture” are important. Positive social environments can reduce genetically based risks and even help to raise intelligence. 3. Assigning meaning to gender in childhood and adolescence is a human universal that influences well-being. Some gender role behavior differences are biologically primed and established prenatally. By the age of 18 to 24 months, children show a deep, vital need to understand and make sense of the same-sex-as-me and the opposite-sex-from-me. Gender identity is much deeper than a mere “set of traits” and runs to the very core of human identity. Not to recognize real differences between males and females can have dangerous consequences. For example, the capacity for pregnancy in adolescent girls places them at special risk for lower education and higher poverty. The aggressive behavior of adolescent boys places them at increased risk for being perpetrators and victims of homicide, suicide, or injuries. 4. A child’s quest for parental approval is the foundation for the emergence of conscience as children learn that certain behaviors are prohibited, permitted, or encouraged. In fact, our sense of right and wrong originates from a biologically primed need to connect with others. 5. And finally, forming a moral identity is an on-going process that becomes increasingly complex as a child matures through childhood and adolescence. It is a process that cannot be left on autopilot. For children, connectedness to adults is a protective factor that helps guide them through difficult times and circumstances. There is no magic in any of this. It is foundational. Families make a difference in providing healthy, stable, connected, contributing individuals who improve all aspects of society, including economic activities. And beyond the general benefit of healthy individuals, we can also look to the benefits of specific skills children learn in families that are transferable to the economic community. Now with these general ideas in mind, let me share some experiences from my life that I hope will prompt you to remember similar circumstances in your lives and consider how family life contributes to productivity. I do so realizing that it is always ¨risky¨ to use personal examples because of their imperfection and perceived lack of professionalism. My mother had a college degree, was an accomplished violinist, and was teaching elementary school when she married my dad. She left teaching in the school system and began teaching in her home as a stay-at-home-mom, her highest aspiration. Because of circumstances within the family of his youth, my father became a principal breadwinner at a very young age. Notwithstanding this significant responsibility, he was determined to get a college education, which he did with my mother’s support. Once he received his undergraduate degree, he entered law school and passed the state bar exam a year prior to graduation from law school. When he graduated the following year, he concluded that rather than pursue the practice of law he would pursue entrepreneurial activities, thus allowing him to include his children more closely in his work. Do you think family life can influence attitudes of children toward the importance of education and its relationship to family? At a very early age I was taught the value of work. The day would begin early in our home, usually before sunrise. Although very young, there were household chores Mother assigned me to do, consistent with my age and capacity. I would sweep, clean, fold, carryout, and mow. My brothers and sisters also participated in these and other activities on the"""
    secular_anchors = """data science data critical thinking machine learning statistics data analysis"""

    religious_anchor_score = compute_speech_topic_score(religious_anchors, religion_topic_vector)
    non_religious_anchor_score = compute_speech_topic_score(secular_anchors, religion_topic_vector)
    religious_text_score = compute_speech_topic_score(biblical_text, religion_topic_vector)
    non_religious_text_score = compute_speech_topic_score(non_religious_text, religion_topic_vector)



    print(f"religious anchor score: {religious_anchor_score}\n non-religious anchor score: {non_religious_anchor_score}\n biblical text score: {religious_text_score}\n non-religious text score: {non_religious_text_score}")
    print("yearly topic scores: " + str(compute_yearly_topic_scores(df, religion_topic_vector))) # expect lower scores for earlier years, swindle mentioned earlier years had more professors'''

    yearly_religion_scores = compute_yearly_topic_scores(df, religion_topic_vector)
    plt.plot(list(yearly_religion_scores.keys()), list(yearly_religion_scores.values()))

    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Neutral Threshold')

    # Add a Linear Regression (Trend) Line
    years = np.array(list(yearly_religion_scores.keys()))
    scores = np.array(list(yearly_religion_scores.values()))

    m, b = np.polyfit(years, scores, 1)
    plt.plot(years, m * years + b, color='blue', linestyle='--', linewidth=1, label='Trend Line')

    plt.xlabel("Year")
    plt.ylabel("Average Religion Topic Score")
    plt.title("Average Religion Topic Score by Year")
    plt.savefig('visualization/yearly_religion_scores.png', dpi=300, bbox_inches='tight')
    plt.close()  # Important: close the figure
    print("Saved yearly_religion_scores.png")

    # Maybe? Like another Topic Vector for something else, would need more anchor words

    # Histogram Comparison of Religious vs Non-Religious Speech
    religious_text = "Ladies and gentlemen, dear friends, I am to greet you on behalf of the Russian Orthodox Church that has over 100 million people in the Russian Orthodox Church. And until now, family values are still very, very strong among Russian people. But as elsewhere in the Christian world, families are under big, big attack. in Russia as in other countries now. And this is the spiritual fight against church and against Christ. And as Dr. Carlson recently mentioned in his speech, that family has the divine origins. Family is not just another institution created by man, by humans. Family is the man. Because man is created by God in his image and in the same way. So man is created as Trinity by God. that has three manifestations, male, female and children. If a is destroyed on Earth, that will inhabit will no be people. If family will be destructed one day, destroyed on earth, the human beings that will be living on earth afterwards, after destruction of the family, will not be the real human beings any longer. When God was creating humankind, He was thinking about it as one living body, one united living body. consisting of families. Man and woman, created them. So when we defending the family, we are actually defending all humankind from self-destruction. Always when the family is destructed it makes humans unhappy on earth. Family is a natural existence for a human being. And there is a divine mechanism for human being in love. And this is the love. Family is a school of love and family is a school of developing love and it's a natural environment, the best environment for humans to exist. When we all together are fighting to protect the natural family, we're actually fighting for the happiness of people on Earth. Only a person that lives in love can understand God and can get to know God. God, whose name is love. So I greet this high, very important meeting from all of my heart. And all our efforts in this direction of protecting family will be blessed by God and will have success of course. So let God bless all of us in our efforts. Thank you for your attention."
    religious_score = compute_word_level_topic_score(religious_text, religion_topic_vector)
    non_religious_text =  """I express my appreciation to Dr. Jesus Hernandez, Dr. Allan Carlson, Dr. Fernando Milanes, Dr. Richard Wilkins, and to those of the organizing committee who extended this invitation and note their considerable contribution to the strength and well-being of families throughout the world. For those of you who don't know me, I would like to briefly introduce myself with some specifics. I grew up in what could be called a traditional family, blessed by the presence and commitment of my mother and father. My parents had six children, three boys and three girls. My father died when I was 20-years-old. A couple of years later, I married Debbie. We have now been married for 34 years. We have eight children, four girls and four boys. Six of our children are married. Our 19th grandchild was born a couple of weeks ago. I am a businessman. However, a few years ago I began devoting much of my time to certain societal matters of interest to me, principally the family. During this time, with a desire to better prepare myself to engage these societal issues, I enrolled at the Kennedy School of Government at Harvard University and there received a master's degree in public administration with a methodological area of concentration in leadership. Since that time, I have become affiliated with several excellent Non-Governmental Organizations that attempt to be an influence for good throughout the world, one of which is United Families International, a co-convening organization for this Congress as identified in your program. I hope you will excuse my sharing so much personal information. I do so because I must ask your further indulgence as I intend to take a risk and draw upon personal experiences relating to family, business, and education as I address you today on the subject of “the Contribution of Family Life to the Productivity of Economies and Companies.” Although economies and companies are sometimes influenced by circumstances beyond their control by things such as natural disasters, the presence or lack of certain natural resources, politics, acts of war, terrorism, etc., the productivity of economies and companies, whether local, regional, national or international, is directly related to the health, strength, and connectedness of the people who are members of the economic unit. Not surprisingly, these issues of health, strength, and connectedness are best nurtured and refined within families. Whether one is providing goods or services, the ability he or she possesses to engage others effectively and perform consistently will influence directly his or her level of production and corresponding contribution to economic growth. This could be called one's “productivity quotient,” or the degree to which one is able to produce more than one consumes. Let me try to say this another way. While much can be said about micro economics and macro economics, about demand curves and supply curves, about elasticity and inelasticity, or about equilibrium and dead weight losses, economic well-being is largely the product of individuals who are stable, net contributors to the world around them. An economic society or company composed entirely of capable, contributing members with high productivity quotients will produce much more than a society or company burdened with many members with low productivity quotients. Please do not misunderstand me. I do not limit the definition of a productivity quotient to monetary measurement. As a businessman, I know that for many, money is the only measure for keeping score. I disagree. Economic well-being cannot be defined merely in terms of the amount of money in one's purse. For example, productivity could include the ideas, social stability, or moral aptitude that one provides for the common good of the economic community. Under this definition, the value of a mother in teaching and nurturing another generation of well-balanced, contributing children may far exceed the value of the business icon's bank account. Public policy considerations, economic or otherwise, that ignore or diminish fundamental commitments to the perpetuation of the health and connectedness of the people are myopic and will ultimately damage society. With this understanding, it would make as much sense to attempt to describe economic productivity without acknowledging the components contributing to the long-term health and connectedness of the people as it would to attempt to describe a cake without identifying the ingredients or the recipe. A little less than a year ago, a group calling itself “The Commission on Children at Risk” presented an excellent work entitled, Hardwired to Connect, The New Scientific Case for Authoritative Communities. The group is composed of 33 children’s doctors, research scientists, and mental health and youth service professionals. The work was published jointly by the YMCA of the United States of America, Dartmouth Medical School, and the Institute for American Values. While its findings have profound implications for virtually every area of human existence, I would like to use them as a backdrop for my comments relating to the role of the family in contributing to economic growth. The report provides empirical evidence that humans are genetically and hormonally driven to connect to other people and to moral meaning. I emphasize that this is “not merely the result of social conditioning, but is instead an intrinsic aspect of the human experience.” The report suggests that this need is best met through what the commission calls “authoritative communities.” These are authoritative social institutions that include children and youth and treat them as ends in themselves, that are warm and nurturing, establish clear limits and expectations, that are multi-generational and have a long-term focus, that transmit a shared understanding of what it means to be a good person, that encourage spiritual and religious development, that teach love of neighbor, and are institutions where the core work is done by non-specialists. It is worth noting that the commission considers the family as “arguably the first and most basic association of civil society, and a centrally important example of what should be an authoritative community.” With that definition, the commission presents ten planks of the new scientific case for authoritative communities. I will not identify them all here, but want to mention five that I consider to be relevant to my topic. You won’t need to remember them specifically, but a general sense of what they convey will be helpful. 1. Nurturing or non-nurturing environments affect gene transcription and the development of brain circuitry. When children are held and loved, they become predisposed at cellular level to pass on good nurturing and physiological resilience to the next generation. In other words, generations can be affected by the nurturing that does or does not occur within a home. 2. Social contexts can alter genetic expression. Both “nature” and “nurture” are important. Positive social environments can reduce genetically based risks and even help to raise intelligence. 3. Assigning meaning to gender in childhood and adolescence is a human universal that influences well-being. Some gender role behavior differences are biologically primed and established prenatally. By the age of 18 to 24 months, children show a deep, vital need to understand and make sense of the same-sex-as-me and the opposite-sex-from-me. Gender identity is much deeper than a mere “set of traits” and runs to the very core of human identity. Not to recognize real differences between males and females can have dangerous consequences. For example, the capacity for pregnancy in adolescent girls places them at special risk for lower education and higher poverty. The aggressive behavior of adolescent boys places them at increased risk for being perpetrators and victims of homicide, suicide, or injuries. 4. A child’s quest for parental approval is the foundation for the emergence of conscience as children learn that certain behaviors are prohibited, permitted, or encouraged. In fact, our sense of right and wrong originates from a biologically primed need to connect with others. 5. And finally, forming a moral identity is an on-going process that becomes increasingly complex as a child matures through childhood and adolescence. It is a process that cannot be left on autopilot. For children, connectedness to adults is a protective factor that helps guide them through difficult times and circumstances. There is no magic in any of this. It is foundational. Families make a difference in providing healthy, stable, connected, contributing individuals who improve all aspects of society, including economic activities. And beyond the general benefit of healthy individuals, we can also look to the benefits of specific skills children learn in families that are transferable to the economic community. Now with these general ideas in mind, let me share some experiences from my life that I hope will prompt you to remember similar circumstances in your lives and consider how family life contributes to productivity. I do so realizing that it is always ¨risky¨ to use personal examples because of their imperfection and perceived lack of professionalism. My mother had a college degree, was an accomplished violinist, and was teaching elementary school when she married my dad. She left teaching in the school system and began teaching in her home as a stay-at-home-mom, her highest aspiration. Because of circumstances within the family of his youth, my father became a principal breadwinner at a very young age. Notwithstanding this significant responsibility, he was determined to get a college education, which he did with my mother’s support. Once he received his undergraduate degree, he entered law school and passed the state bar exam a year prior to graduation from law school. When he graduated the following year, he concluded that rather than pursue the practice of law he would pursue entrepreneurial activities, thus allowing him to include his children more closely in his work. Do you think family life can influence attitudes of children toward the importance of education and its relationship to family? At a very early age I was taught the value of work. The day would begin early in our home, usually before sunrise. Although very young, there were household chores Mother assigned me to do, consistent with my age and capacity. I would sweep, clean, fold, carryout, and mow. My brothers and sisters also participated in these and other activities on the"""
    non_religious_score = compute_word_level_topic_score(non_religious_text, religion_topic_vector)


    religious_score = compute_word_level_topic_score(religious_text, religion_topic_vector)
    non_religious_score = compute_word_level_topic_score(non_religious_text, religion_topic_vector)

    # Side-by-side histograms (easier to compare)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(religious_score, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_title('Religious Speech By Archbishop Dmitry Smirnov\nWord-Level Religion Scores')
    axes[0].set_xlabel('Religion Topic Score')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(religious_score), color='red', linestyle='--', label=f'Mean: {np.mean(religious_score):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, 0.5)  # Set same x-axis limits for better comparison

    axes[1].hist(non_religious_score, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_title('Non-Religious Speech By Craig A. Cardon\nWord-Level Religion Scores')
    axes[1].set_xlabel('Religion Topic Score')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(non_religious_score), color='red', linestyle='--', label=f'Mean: {np.mean(non_religious_score):.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.5, 0.5)  # Set same x-axis limits for better comparison

    plt.tight_layout()
    plt.savefig('visualization/histogram_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved histogram_comparison.png")

    print("done")

    # Different Aggregation Methods: Box Plot Comparison of Religious vs Non-Religious Speech

