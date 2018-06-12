from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from sklearn import metrics

class Model():
    def __init__(self):
        self.PATH = "data/dog-breed-identification/"
        self.sz=299
        self.arch=resnext101_64
        self.bs=58

        label_csv = f'{self.PATH}labels.csv'
        n = len(list(open(label_csv)))-1
        self.val_idxs = get_cv_idxs(n)
        
        self.data = self.get_data(self.sz, self.bs) #data generator for batch size=64, image size=64x64
        self.model = ConvLearner.pretrained(self.arch, self.data)
        self.model.load('img_sz_299_resnext101_64_wo_unfreeze_cycle_1_2_eta_1e-2_date_2018-05-20 14:27:16.434806')
        print('Model loaded')
        
        self.clasess_ru = ['аффенпинчер',
 "афганская борзая",
 "африканская охотничья собака",
 'эрдельтерьер',
 'американский стаффордширский терьер',
 'аппенцеллер',
 'австралийский терьер',
 'басенджи',
 'бассет',
 'бигль',
 'бедлингтон-терьер',
 "бернский зенненхунд",
 "черно-подпалый кунхаунд",
 'бленхейм спаниель',
 'ищейка',
 'блутик',
 'колли',
 'бордер терьер',
 'борзая',
 'бульдожка',
 'фландрский бувье',
 "боксер",
 'брабансон гриффон',
 'бриар',
 'Бриттани спаниель',
 "бычий мастиф",
 'Керн',
 'кардиган',
 'чесапик Бэй ретривер',
 'чихуахуа',
 'Чоу',
 'кламбер',
 'кокер-спаниель',
 "колли",
 'кудрявый ретривер',
 'денди динмонт',
 'красный волк',
 'Динго',
 'доберман',
 'английский фоксхаунд',
 'английский сеттер',
 'английский спрингер',
 'Энтлебухер зенненхунд',
 "эскимосская собака",
 'прямошерстный ретривер',
 "французский бульдог",
 "немецкая овчарка",
 "немецкий короткошерстный пойнтер",
 'ризеншнауцер',
 "золотистый ретривер",
 'сеттер',
 'дог',
 'великий Пиренеях',
 'большой швейцарский зенненхунд',
 'грюнендаль',
 'ibizan hound',
 "ирландский сеттер",
 'ирландский терьер',
 'ирландский водяной спаниель',
 "ирландский Волкодав",
 'левретка',
 'японский спаниель',
 'Акита',
 'келпи',
 'Керри-Блю-терьер',
 "комондор",
 'кувас',
 "лабрадор ретривер",
 'лейкленд терьер',
 'леонберг',
 'лхаса',
 "маламут",
 'малинуа',
 'слушаю',
 "мексиканская безволосая",
 'цвергпинчер',
 "миниатюрный пудель",
 'цвергшнауцер',
 "ньюфаундленд",
 'Норфолк терьер',
 'норвежский элкхаунд',
 'Норвич терьер',
 'староанглийская овчарка',
 'собаки',
 'папийон',
 'пекинес',
 'пемброк',
 'шпиц',
 'мопс',
 'рыжуля',
 "родезийский риджбек",
 'ротвейлер',
 'сенбернар',
 'салюки',
 'самоед',
 'шипперке',
 'скотч-терьер',
 'дирхаунд',
 'силихем терьер',
 'шелти',
 'ши-тцу',
 "Сибирский хаски",
 'шелковистый терьер',
 'мягкошерстный пшеничный терьер',
 "стаффордширский бультерьер",
 'пудель',
 "стандартный шнауцер",
 "сассекс-спаниель",
 "тибетский мастиф",
 'тибетский терьер',
 'пудель',
 'той-терьер',
 'выжла',
 'Уокер хаунд',
 'собака',
 'вельш-спрингер-спаниель',
 'Вест хайленд уайт терьер',
 'уиппет',
 'жесткошерстный фокстерьер',
 'йоркширский терьер']

    def get_data(self, sz, bs):
        tfms = tfms_from_model(self.arch, self.sz, aug_tfms=transforms_side_on, max_zoom=1.1)
        data = ImageClassifierData.from_csv(self.PATH, 'train', f'{self.PATH}labels.csv', test_name='test' ,tfms=tfms, 
                                    bs=self.bs, val_idxs=self.val_idxs, suffix='.jpg')
        return data if sz>300 else data.resize(340, 'tmp')

    def identify_dog_breed(self, file_path):
        trn_tfms, val_tfrms = tfms_from_model(self.arch, self.sz)
        img = trn_tfms(open_image(file_path))
        preds = self.model.predict_array(img[None])
        return self.clasess_ru[np.argmax(np.exp(preds))]