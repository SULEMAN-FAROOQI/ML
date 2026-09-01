import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Suppress INFO, WARNING, and ERROR logs from TF/absl

import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout  # type: ignore
from scikeras.wrappers import KerasClassifier

faqs = """About Mammals and Terrestrial Vertebrates
What is the primary diet of the African Savannah Elephant?
African Savannah Elephants are strict herbivores. An adult elephant consumes between 100 to 150 kg of vegetation daily, consisting mainly of grasses, tree bark, roots, soft leaves, and seasonal fruits.

Where do African Savannah Elephants live?
They inhabit diverse ecosystems across Sub-Saharan Africa, including open savannas, grasslands, mopane woodlands, and semi-arid deserts.

What do Giant Pandas eat, and why is their diet unique?
Giant Pandas are specialized herbivores whose diet is over 99% bamboo. Because bamboo is low in nutritional value, an adult panda must spend up to 12 hours a day feeding, consuming 12 to 38 kg of bamboo stems, leaves, and shoots daily.

What is the natural habitat of the Giant Panda?
Giant Pandas live in high-altitude, cool, moist bamboo forests in the mountainous regions of central China, specifically in Sichuan, Shaanxi, and Gansu provinces.

What feeding strategy do Bengal Tigers use?
Bengal Tigers are apex carnivores and solitary stalk-and-ambush predators. They hunt large-to-medium mammals such as chital (spotted deer), sambar, wild boar, and water buffalo.

What is the native habitat of the Bengal Tiger?
Their habitat spans tropical rainforests, mangroves (such as the Sundarbans), moist deciduous forests, and alluvial grasslands across India, Nepal, Bhutan, and Bangladesh.

How does the diet of a Red Panda differ from that of a Giant Panda?
While both rely heavily on bamboo, Red Pandas are omnivorous. In addition to bamboo leaves and shoots, they supplement their diet with berries, fruits, blossoms, acorns, bird eggs, and small insects.

What environment does the Red Panda inhabit?
Red Pandas inhabit temperate coniferous and deciduous mountain forests with dense bamboo understories in the Himalayas, across Nepal, India, Bhutan, Myanmar, and southwestern China.

About Aquatic and Marine Life
What is the main food source for Blue Whales?
Blue Whales are specialized filter feeders that feed almost exclusively on tiny marine crustaceans called krill. During peak feeding season, a single adult Blue Whale can consume up to 4 tons (3,600 kg) of krill per day.

Where are Blue Whales found throughout the year?
Blue Whales inhabit all of the world's major oceans except the Arctic. They migrate seasonally between cold, nutrient-rich polar feeding grounds in the summer and warm, tropical breeding waters near the equator in the winter.

What do Great White Sharks eat as they mature?
Juvenile Great White Sharks feed primarily on fish, rays, and smaller ocean creatures. As they mature into adults, their diet shifts to marine mammals, including harbor seals, sea lions, elephant seals, and small whales.

What habitat do Great White Sharks prefer?
They live in coastal and offshore waters with temperatures between 12°C and 24°C, frequently found near seal colonies along the coasts of California, South Africa, Australia, and the Mediterranean Sea.

Diet and Habitat Encyclopedia: Animal Kingdom
Frequently Asked Questions and Knowledge Base

Section 1: Terrestrial Mammals

What is the habitat of the Bengal Tiger?
The Bengal tiger primarily inhabits the tropical moist evergreen forests, tropical dry forests, mangrove forests of the Sundarbans, and subtropical alluvial grasslands of the Indian subcontinent. They require dense vegetation, proximity to water sources, and an abundant prey base to sustain their territorial needs.

What does the Bengal Tiger eat?
As apex obligate carnivores, Bengal tigers feed mainly on large ungulates such as ungulate deer including chital, sambar, and barasingha, as well as wild boar, gaur, and water buffalo. In times of scarcity, they may target smaller prey like porcupines, hares, and birds.

What is the preferred natural environment of the African Bush Elephant?
African bush elephants are highly adaptable and occupy diverse habitats ranging from open savannahs, grasslands, and miombo woodlands to semi-arid deserts and dense marshes across Sub-Saharan Africa. They depend heavily on vast home ranges with reliable access to fresh drinking water and trees for shade and foraging.

What is the daily diet of an African Bush Elephant?
African bush elephants are megaherbivores that practice generalist browsing and grazing. They consume hundreds of pounds of vegetation daily, including native grasses, tree bark, leaves, roots, fruits, and branches. Their diet shifts seasonally depending on the availability of green forage versus woody vegetation.

Where do Giant Pandas live in the wild?
Giant pandas live exclusively in the high-altitude bamboo forests of mountain ranges in central China, predominantly across Sichuan, Shaanxi, and Gansu provinces. These humid, cool cloud forests provide dense bamboo canopies that shelter them from extreme climate variations.

What is the primary food source of the Giant Panda?
Despite belonging to the order Carnivora, the giant panda's diet is over ninety-nine percent bamboo. They consume various parts of the plant including stems, shoots, and leaves from species like arrow bamboo and umbrella bamboo. Because bamboo is fibrous and low in nutrients, pandas spend up to twelve hours a day eating to meet their metabolic demands.

Where are Koalas found and what constitutes their habitat?
Koalas are endemic to eastern and southeastern Australia, residing in open eucalypt woodlands where eucalyptus trees dominate the canopy. They are arboreal mammals spending most of their lives high up in tree branches that supply both food and shelter.

Why do Koalas have such a specialized diet?
Koalas are dietary specialists that feed almost exclusively on the foliage of specific eucalyptus species. Eucalyptus leaves are extremely fibrous, low in nutrition, and toxic to most other mammals. Koalas have evolved specialized liver enzymes to neutralize these toxins and an elongated cecum to digest the fibrous material through bacterial fermentation.

Where do Snow Leopards live?
Snow leopards inhabit the rugged, high-mountain alpine and subalpine zones across Central and South Asia, including the Himalayas, Pamirs, and Altai mountains. They prefer steep, rocky terrain, ridges, cliffs, and ravines that offer optimal camouflage and vantage points for hunting.

What do Snow Leopards hunt in their mountain habitats?
Snow leopards are opportunistic carnivores that prey primarily on wild caprids such as blue sheep, ibex, markhor, and argali. They also supplement their diet with smaller creatures like marmots, pikas, hares, and game birds when large prey is scarce.

What is the natural habitat of the Red Kangaroo?
Red kangaroos inhabit the arid and semi-arid inland regions of Australia, including open plains, shrublands, grasslands, and desert environments with sparse tree cover.

What do Red Kangaroos eat?
Red kangaroos are herbivorous grazers that feed predominantly on green grasses and young forb species. They can survive on low-quality forage and derive much of their hydration from the plants they consume, enabling them to thrive in harsh drought conditions.

Section 2: Marine Mammals and Aquatic Life

What habitat does the Blue Whale occupy?
Blue whales are pelagic marine mammals found in ocean environments worldwide, ranging from polar feeding grounds to tropical calving regions. They travel through deep oceanic waters, following major currents and nutrient-rich upwelling zones.

What is the main component of a Blue Whale's diet?
Despite their massive size, blue whales feed almost exclusively on tiny marine crustaceans known as krill. Using their baleen plates, they filter massive quantities of seawater, consuming up to four tons of krill daily during peak feeding seasons.

What environment do Sea Otters inhabit?
Sea otters inhabit temperate coastal waters of the Northern Pacific Ocean. They prefer shallow coastal environments, rocky reefs, and dense kelp forests where they can forage along the sea floor and wrap themselves in giant kelp to anchor while resting.

What is the diet of a Sea Otter?
Sea otters are carnivorous marine predators that feed on benthic invertebrates including sea urchins, abalone, clams, mussels, crabs, and snails. By consuming herbivorous sea urchins, otters prevent the destruction of kelp forests, serving as a critical keystone species.

Where do Bottlenose Dolphins live?
Bottlenose dolphins occupy temperate and tropical waters worldwide, inhabiting coastal bays, estuaries, lagoons, and continental shelves, as well as deep offshore waters.

What do Bottlenose Dolphins eat?
Bottlenose dolphins are opportunistic carnivores whose diet consists primarily of schooling fish, squids, and crustaceans. They employ complex cooperative hunting strategies, such as mud-ring feeding and fish-whacking, to trap prey.

Section 3: Birds and Avian Species

Where do Emperor Penguins live?
Emperor penguins are native to Antarctica, living on pack ice, ice shelves, and surrounding sub-Antarctic waters. They require stable fast ice attached to the coast for breeding colonies and moulting grounds.

What is the dietary composition of Emperor Penguins?
Emperor penguins feed on marine organisms during deep dives into icy ocean waters. Their diet consists primarily of fish such as Antarctic silverfish, crustaceans like krill, and various cephalopod species like squid.

What is the habitat of the Peregrine Falcon?
Peregrine falcons are cosmopolitan birds of prey found on every continent except Antarctica. They nest on high cliffs, mountain ledges, river bluffs, and increasingly on urban skyscrapers that simulate natural vertical cliffs.

How do Peregrine Falcons obtain their food?
Peregrine falcons are aerial predators that hunt other birds, including pigeons, doves, waterfowl, and songbirds. They capture prey in mid-air by stooping—diving at extreme speeds exceeding two hundred miles per hour to strike prey with locked talons.

Where can the Harpy Eagle be found?
Harpy eagles reside in the upper canopy of undisturbed lowland tropical rainforests in Central and South America, particularly in the Amazon basin.

What do Harpy Eagles feed on?
Harpy eagles are top-tier avian predators that specialize in arboreal mammals. Their primary prey includes tree-dwelling sloths, howler monkeys, capuchin monkeys, opossums, and large forest birds like macaws.

Section 4: Reptiles and Amphibians

What is the habitat of the Komodo Dragon?
Komodo dragons are endemic to several Indonesian islands, including Komodo, Rinca, Flores, and Gili Motang. They inhabit tropical dry forests, arid savannas, open woodlands, and coastal mangroves.

What does the Komodo Dragon eat?
Komodo dragons are hypercarnivores and scavengers. Juvenile dragons consume insects, geckos, and small rodents in trees, while adults hunt large mammals like timor deer, wild pigs, water buffalo, and goats using ambush techniques paired with toxic venomous bite delivery.

Where do Poison Dart Frogs live?
Poison dart frogs live in the humid, tropical rainforests of Central and South America. They thrive in leaf litter, under rocks, and near moist streams or within epiphytic bromeliads high in the forest canopy.

What is the diet of Poison Dart Frogs and how does it affect their toxicity?
Poison dart frogs feed on small invertebrates such as ants, termites, beetles, and mites. They derive their alkaloid toxins directly from their diet; ants and mites contain chemical compounds that the frogs sequester into their skin glands as a chemical defense mechanism.

Section 5: Arthropods and Insects

What is the habitat of the Leafcutter Ant?
Leafcutter ants live in underground nest systems built in the tropical rainforests and savannas of Central and South America. Their subterranean colonies can extend several meters deep into the soil.

Do Leafcutter Ants eat the leaves they harvest?
No, leafcutter ants do not directly eat the leaves they cut. Instead, they chew the foliage into pulp to cultivate a specialized fungus garden within their nest. The ants feed exclusively on the nutrient-rich structures produced by this cultivated fungus.

Where do Monarch Butterflies reside during their lifecycle?
Monarch butterflies inhabit fields, meadows, roadsides, and coastal areas across North America where milkweed plants grow. During winter, eastern populations migrate thousands of miles to overwinter in dense oyamel fir forests in central Mexico.

What is the diet of a Monarch Butterfly caterpillar versus an adult?
Monarch caterpillars are obligate herbivores that feed strictly on milkweed foliage, absorbing cardenolide toxins that make them unpalatable to predators. Adult monarchs are nectarivores, feeding on flower nectar from a wide range of flowering plants to power their long-distance migrations."""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])  # We provide it in a list because tokenizer can take multiple lists.
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding/reserved index 0
# print(vocab_size)

input_sequences = []
for sentence in faqs.split("\n"):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]  # Converts sentences to numeric representation, [0] unwraps the outer list.

    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[: i + 1])

max_len = max([len(x) for x in input_sequences])
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="pre")

x = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]

y = to_categorical(y, num_classes=vocab_size)  # One Hot Encoding

# print(x.shape)
# print(y.shape)

TIMESTEPS = max_len - 1  # length of each input sequence (x.shape[1])


def build_model(meta):

    m = Sequential()

    m.add(Embedding(vocab_size, 180, input_shape=(TIMESTEPS,)))

    m.add(LSTM(128, return_sequences=True)) 
    m.add(Dropout(0.2))

    m.add(LSTM(128, return_sequences=True))  
    m.add(Dropout(0.2))

    m.add(LSTM(64))                          
    m.add(Dropout(0.2))

    m.add(Dense(vocab_size, activation="softmax"))

    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return m


m = KerasClassifier(model=build_model, epochs=80, batch_size=33, verbose=1, validation_split=0.3)
m.fit(x, y)

# Testing:

text = """Red Pandas inhabit temperate coniferous and deciduous mountain forests with dense bamboo understories in the Himalayas, across Nepal, India, Bhutan, Myanmar, and southwestern """

tokkened_text = tokenizer.texts_to_sequences([text])[0]
padded_tokkened_texts = pad_sequences([tokkened_text], maxlen=max_len - 1, padding="pre")

prediction = m.predict(padded_tokkened_texts)  # already decoded back to the class label
position = np.argmax(prediction[0])

for word, index in tokenizer.word_index.items():
    if index == position:
        text = text + "" + word
        print(text)

text = "Poison dart frogs feed on small invertebrates such as ants, termites, beetles, and mites. They derive their alkaloid toxins directly from their diet; ants and mites contain chemical compounds that the frogs sequester into their skin glands as a"

''' Poison dart frogs feed on small invertebrates such as ants, termites, beetles, and mites. They derive their alkaloid toxins directly from their diet; ants and mites contain chemical compounds that the frogs sequester into their skin glands as a chemical defense mechanism '''

for i in range(3):
    tokkened_text = tokenizer.texts_to_sequences([text])[0]
    padded_tokkened_texts = pad_sequences([tokkened_text], maxlen=max_len - 1, padding="pre")

    prediction = m.predict(padded_tokkened_texts)  # already decoded back to the class label
    position = np.argmax(prediction[0])

    for word, index in tokenizer.word_index.items():
        if index == position:
            text = text + " " + word
            print(text)
            break