MEANS = 'after before cow deity farm grain hate horse king love money person rain soul tool war'.split()
MEAN_DIM = len(MEANS)
GRAMS = 'verb noun adj prep det conj modal adv'.split()
GRAM_DIM = len(GRAMS)

import random

# "How a man of the country told to him of a marvellous giant, and how he fought and conquered him"

class Writer:
    def __init__(self):
        pass

    def make_determiner(self):
        return random.choice('a the my your every many-a no such-a'.split())

    def make_countable_noun(self):
        return random.choice((
            #'cow god sun fruit seed worm crab dance veil ship wind penalty'+
            'land thing totality knowledge secret flood journey peace stone'+
            ' slab wall temple sanctuary copper threshold residence king man'+
            ' foundation brick core kiln sage plan league city palm garden'+
            ' lowland area tablet box lock bronze fastening opening'+
            ' lapis-lazuli appearance hero bull front leader companion'+
            ' net protector flood-wave offspring perfection cow pass water-well'+
            ' flank ocean sun region name day birth god human model enclosure'+
            ' head rival weapon decree day night shepherd girl mother warrior'+
            ' bride complaint heaven lord memorial match hand clay wilderness'+
            ' silence hair garment gazelle watering-hole animal thirst water'+
            ' trapper home heart face core father meteorite pit trap grasp'+
            ' woman robe way post depth arm energy'+
            ' bosom charm knee ear oiwer favor friend finery lyre drum couch'+
            ' sheet feeling youth thought mind dream start sky populace foot'+
            ' baby comrade gate chamber axe adviser settled-living'+
            #
            ' country giant person year child common attendant lodging'+
            ' mountain end hundred lust wife cousin blood pity revenge'+
            ' conqueror mischief realm lady fellow fire treasure case tent'+
            ' horse harness evening-song pilgrimage shield hill widow hand'+
            ' grave lamentation devil resistance world navel tyrant treaty'+
            ' coat stone beard love salvation supper message word crest limb'+
            ' man limb broach bird sight compassion death soul glutton club'+
            ' crown ground belly gut ribcage help comfort beach dagger head'+
            ' spear barbican number gown host worship morning battle'+
            ' valley messenger emperor part slaughter town body prize'+
        '').split())
    def make_uncountable_noun(self):
        return random.choice((
            #'rain blood milk honey love hate battle grain prayer clay mourning lies skystuff mountainrock'+
            'time exhaustion toil hardship strength life mankind grass fear'+
            ' duty'+
            #
            ' iron'+
        '').split())

    def make_preposition(self):
        return random.choice('of from above below named-after beside inside'.split())
    def make_prepositional_adverb(self):
        return random.choice('with like against born-from for/to on'.split())

    def make_noun_modifier(self):
        return random.choice([
            lambda:'{} {}'.format(self.make_preposition(), self.make_noun_phrase()),
        ])()
    def make_noun_phrase(self):
        base = random.choice([
            lambda:'{} {}'.format(self.make_determiner(), self.make_countable_noun()), 
            lambda:'{}'.format(self.make_uncountable_noun()), 
        ])()
        if random.random()<0.05:
            base = '{} {}'.format(base, self.make_noun_modifier())
        if random.random()<0.05:
            base = random.choice([
                lambda:'{} that {} {}'.format(base, self.make_noun_phrase(), self.make_trans_verb()),
                lambda:'{} that {}'.format(base, self.make_verb_phrase()),
            ])()
        return base

    def make_trans_verb(self):
        return random.choice((
            #'praised worshipped fed ate conquered raised loved feared took made spat-at moulded sold bought filled granted'+
            'saw revealed taught granted discovered pushed built carved'+
            ' examined equaled grasped walked enclosed found opened undid'+
            ' read suffered gored walked trusted destroyed opened dug crossed'+
            ' explored sought reached restored teemed was-compared-to'+
            ' prepared raised stood left heard implored created found washed'+
            ' pinched threw was-endowed-with billowed wore ate jostled-at'+
            ' slaked opposed addressed planted filled released gave exposed'+
            ' drew-near-to restrained took entertained unclutched sated'+
            ' gazed-at shouted led changed showed-off enlarged lifted'+
            ' embraced laid-down saved collected possessed interpreted'+
            ' recounted'+
            #
            ' slayed devoured sustained destroyed led ravished rescued' +
            ' shrieked cried fulfilled brought understood commanded prepared'+
            ' armed took ascended wrung answered forced slit treated gave'+
            ' approached decorated sent accomplished gnawed baked broached'+
            ' beheld bled hailed wielded dressed smote hit carved'
            ' cut-off spilled threw caught crushed called-to weltered'+
            ' wallowed rolled kept loosed bound fetched knew thanked'+
            ' distributed blessed built removed erected entered burned'+
            ' yielded'+
        '').split())
    def make_intrans_verb(self):
        return random.choice((
            #'arrived left died wept fell ripened waited rotted groaned slept danced awoke erupted'+
            'gleamed raged teemed paled grew-up arrived groaned galloped'+
            ' strutted stood played slept budged assembled thronged clustered'+
            ' appeared passed'+
            #
            ' rode departed waited sat spoke turned started kneeled arose'+
            ' fell'+
        '').split())
    def make_verb_phrase(self):
        base = random.choice([
            lambda:'{}'.format(self.make_intrans_verb()),
            lambda:'{} {}'.format(self.make_trans_verb(), self.make_noun_phrase()), 
            lambda:'was {} {}'.format(self.make_prepositional_adverb(), self.make_noun_phrase()), 
        ])()
        while random.random()<0.10:
            base = random.choice([
                lambda:'{} {} {}'.format(base, self.make_prepositional_adverb(), self.make_noun_phrase()),
                lambda:'{} {} {}'.format(self.make_prepositional_adverb(), self.make_noun_phrase(), base),
            ])()

        return base

    def make_sentence(self):
        base = random.choice([
            lambda:'{} {}'.format(self.make_noun_phrase(), self.make_verb_phrase()),
        ])()
        if random.random()<0.10:
            base = random.choice([
                lambda:'{} so that {}'.format(base, self.make_sentence()),
                lambda:'{} said that {}'.format(self.make_noun_phrase(), base),
                lambda:'{} but {}'.format(self.make_sentence(), base),
            ])()
        return base

    def make_vocative(self):
        base = random.choice([
            lambda:'{}'.format(self.make_countable_noun()),
            lambda:'{}'.format(self.make_uncountable_noun()),
        ])()
        if random.random()<0.10:
            base = random.choice([
                lambda:'{} that {} {}'.format(base, self.make_noun_phrase(), self.make_trans_verb()),
                lambda:'{} that {}'.format(base, self.make_verb_phrase()),
            ])()
        return base
    def make_utterance(self):
        base = self.make_sentence()
        if random.random()<0.10:
            base = random.choice([
                lambda:'O {}, {}'.format(self.make_vocative(), base),
                lambda:'and {}'.format(base),
                lambda:'then {}'.format(base),
                lambda:'when {}, {}'.format(base, self.make_sentence()),
                lambda:'unless {}, {}'.format(base, self.make_sentence()),
                lambda:'fearing that {}, {}'.format(base, self.make_sentence()),
                lambda:'because of {}, {}'.format(self.make_noun_phrase(), base),
            ])()
        return base

W = Writer()

def make_paragraph(L=16):
    text = ''
    while len(text.split())<L:
        text += '{}.  '.format(W.make_utterance())
    return text

print()
for _ in range(20):
    print(make_paragraph()) 
print()

#ENGLISH = {
#    'cattle'        : ('cow'   , 'cow'  , 'noun'),
#    'livestock'     : ('farm'  , 'cow'  , 'noun'),
#    'feast'         : ('meat'  , 'cow'  , 'noun'),
#    'feast'         : ('meat'  , 'cow'  , 'noun'),
#    #
#    'scythe'        : ('farm'  , 'tool' , 'noun'),
#    'spear'         : ('war'   , 'tool' , 'noun'),
#    'crown'         : ('king'  , 'tool' , 'noun'),
#    'chariot'       : ('horse' , 'tool' , 'noun'),
#    #
#    'death'         : ('after' , 'soul' , 'noun'),
#    'birth'         : ('before', 'soul' , 'noun'),
#    ####
#    'died'          : ('after' , 'soul' , 'verb'),
#    'was born'      : ('before', 'soul' , 'verb'),
#}
