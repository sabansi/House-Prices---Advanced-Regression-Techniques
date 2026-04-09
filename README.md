# House Prices - Advanced Regression Techniques

### კონკურსის მიმოხილვა

Kaggle House Prices კონკურსის მიზანია საცხოვრებელი სახლების გაყიდვის ფასის პროგნოზირება — ისეთი მახასიათებლის საფუძველზე, როგორებიცაა ფართობი, სარდაფი, გარაჟი, სამეზობლო და სხვა. ეს ამოცანა ფასდება RMSLE (Root Mean Squared Log Error) მიხედვით.


<img width="986" height="351" alt="saleprice_distribution" src="https://github.com/user-attachments/assets/0187331d-c3b1-4552-9212-3d49ed33c685" />

---

## რეპოზიტორიის სტრუქტურა

```
House-Prices---Advanced-Regression-Techniques/
│
├── model_experiment.ipynb     ← EDA, cleaning, feature engineering, feature selection, ექსპერიმენტები
├── model_inference.ipynb      ← საუკეთესო მოდელის ჩატვირთვა Model Registry-იდან, submission 
└── README.md
```

---

## ფაილების აღწერა


| ფაილი                    | აღწერა                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `model_experiment.ipynb` | მთავარი notebook — data cleaning, feature engineering, feature selection და მოდელების ექსპერიმენტები              |
| `model_inference.ipynb`  | MLflow Model Registry-იდან საუკეთესო მოდელის ჩატვირთვა, Kaggle-ის ტესტ სეტზე პროგნოზი და submission.csv გენერაცია |


---

## Data Cleaning

### 1. გამოტოვებული მნიშვნელობების ანალიზი

გავაანალიზე თითოეული სვეტის missing value-ების პროცენტი.

<img width="989" height="590" alt="missing_values" src="https://github.com/user-attachments/assets/d52f0be5-ce81-4aac-b66e-c6f965b799d1" />


### 2. Simple Imputation

გამოვიყენე მარტივი შევსების სტრატეგია:

- **რიცხვითი სვეტები** შევავსე **მედიანით** — მედიანა უფრო მდგრადია outlier-ების მიმართ ვიდრე საშუალო, საშუალო შეიძლება არაზუსტად იყოს გადახრილი და მედიანა შედარებით ზუსტ მონაცემს გვიჩვენებს
- **კატეგორიული სვეტები** შევავსე **მოდით** — ყველაზე ხშირი მნიშვნელობით

fill value-ები გამოვითვალე მხოლოდ X_train-ზე და შემდეგ გამოვიყენე X_test-ზეც data leakage თავიდან ასაცილებლად.

### 3. Outlier-ების მოცილება

`GrLivArea` vs `SalePrice`-ის scatter plot-ზე გამოვავლინე 2 extreme outlier — ძალიან დიდი სახლები, რომლებიც უჩვეულოდ დაბალ ფასად გაიყიდა. ეს სახლები სავარაუდოდ სპეციალური პირობებით გაიყიდა და მოდელს დააბნევდა.


<img width="700" height="391" alt="outliers" src="https://github.com/user-attachments/assets/e3125245-c8a1-4002-9e4d-d3b585bf8916" />

---

## Feature Engineering

### ახალი features

შევქმენი შემდეგი ახალი სვეტები:

- **TotalSF** = სარდაფი + 1-ლი სართული + მე-2 სართული — სახლის საერთო ფართობი
- **HouseAge** = გაყიდვის წელი − აშენების წელი
- **RemodAge** = გაყიდვის წელი − რემონტის წელი
- **TotalBath** = სრული აბაზანა + ნახევარი × 0.5 + სარდაფის აბაზანები
- **HasGarage**, **HasFireplace**, **HasPool**, **HasPorch** — ბინარული სვეტები, არსებობს თუ არა ობიექტი

### კატეგორიული ცვლადები

#### OneHotEncoder

გამოვიყენე **დაბალი კარდინალობის** კატეგორიულ სვეტებზე (5-ზე ნაკლები უნიკალური მნიშვნელობა):
`MSZoning`, `CentralAir`, `SaleType`, `Foundation` და სხვა — სულ 30 სვეტი.

#### Ordinal Encoding

ხარისხის სვეტებს (`ExterQual`, `KitchenQual`, `BsmtQual` და სხვა) აქვთ რიცხვითი მნიშვნელობები კატეგორიულ ცვლადებზე:
`Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0`

მოდელი პირდაპირ ხედავს, რომ Ex > Gd > TA...

#### WOE Encoding

გამოვიყენე **მაღალი კარდინალობის** სვეტებზე — `Neighborhood`, `Exterior1st`, `Exterior2nd`.

OHE-ს ნაცვლად WOE-მ ეს სვეტები 1 სვეტად შეკუმშა, სადაც თითოეული სამეზობლოს მნიშვნელობა ასახავს მის კავშირს ფასთან:

- **დადებითი WOE** → ძვირი სამეზობლო (მაღალი ფასი)
- **უარყოფითი WOE** → იაფი სამეზობლო (დაბალი ფასი)

---

## Feature Selection

### 1. კორელაციის ფილტრი (Threshold = 0.85)

ამოვიღე სვეტები, რომლებს შორის კორელაცია 0.85-ს აღემატებოდა. სულ **10 სვეტი** ამოვიღე:

მაგალითად:

- `GarageArea` ↔ `GarageCars` — დიდი გარაჟი = მეტი მანქანა 
- `TotalSF` ↔ `GrLivArea` — TotalSF შეიცავს GrLivArea-ს
- `HouseAge` ↔ `YearBuilt`

165 სვეტი დარჩა კორელაციის ფილტრის შემდეგ.

![correlation_matrix](https://github.com/user-attachments/assets/dffba485-a993-4fb5-8c21-cfbdfa46b461)


### 2. RFE

LinearRegression-ის გამოყენებით შევარჩიე **50 ყველაზე მნიშვნელოვანი სვეტი**. RFE თანდათანობით შლის ყველაზე სუსტ სვეტებს, სანამ სასურველ რაოდენობას არ მიაღწევს.

### 3. IV — Threshold = 0.02

IV ზომავს სვეტის პროგნოზულ ძალას. თუ **IV < 0.02**, სვეტი პრაქტიკულად უსარგებლოა.

სულ **28 სვეტი** ამოვიღე IV ფილტრით — ძირითადად იშვიათი კატეგორიები როგორებიცაა `RoofMatl_Metal`, `Utilities_NoSeWa`.

საბოლოოდ დარჩა **22 სვეტი**.

![iv_scores](https://github.com/user-attachments/assets/466a8f5c-0b6c-404e-a299-067acdd44c06)


### Feature Set-ების შედარება (Linear Regression-ზე)


| Feature Set  | სვეტები | test_rmse | ანალიზი                                      |
| ------------ | ------- | --------- | -------------------------------------------- |
| all_features | 165     | 0.2680    | Overfitting — val და test შორის დიდი სხვაობა |
| rfe_features | 50      | 0.1716    | საუკეთესო                                    |
| iv_features  | 22      | 0.1748    | მსგავსი შედეგი, ნაკლები სვეტებით             |


---

## ტრენინგი და ექსპერიმენტები

ყველა ექსპერიმენტი დავლოგე MLflow-ში DagsHub-ზე. თითოეული მოდელისთვის დავლოგე:

- `train_rmse`, `val_rmse`, `test_rmse`
- `train_r2`, `val_r2`, `test_r2`
- ყველა ჰიპერპარამეტრი
- დატრენინგებული მოდელი

---

### Linear Regression (Baseline)


| Feature Set        | val_rmse | test_rmse | ანალიზი                                          |
| ------------------ | -------- | --------- | ------------------------------------------------ |
| all_features (165) | 0.1332   | 0.2680    | **Overfitting** — val ბევრად უკეთესია ვიდრე test |
| rfe_features (50)  | 0.1544   | 0.1716    | კარგი ბალანსი                                    |
| iv_features (22)   | 0.1589   | 0.1748    | კარგი, ნაკლები სვეტებით                          |


---

### Ridge Regression (L2 Regularization)

alpha პარამეტრი აკონტროლებს regularization-ის სიძლიერეს.


| alpha | val_rmse | test_rmse | ანალიზი                                  |
| ----- | -------- | --------- | ---------------------------------------- |
| 0.01  | 0.1716   | 0.1716    | საუკეთესო                                |
| 1.0   | 0.1524   | 0.1940    | val უმჯობესდება, test უარესდება          |
| 500.0 | 0.1764   | 0.1916    | **Underfitting** — ძალიან ძლიერი penalty |


---

### Lasso Regression (L1 Regularization)

შეუძლია კოეფიციენტები ზუსტად ნულამდე შეამციროს.


| alpha  | val_rmse | test_rmse | zeroed features | ანალიზი                                |
| ------ | -------- | --------- | --------------- | -------------------------------------- |
| 0.0001 | 0.1531   | 0.1702    | 4               | საუკეთესო — სუსტი regularization       |
| 0.0005 | 0.1532   | 0.1769    | 21              | კარგი ბალანსი                          |
| 0.1    | 0.2362   | 0.2500    | 48              | **Underfitting** — 48/50 სვეტი ნულდება |


---

### Decision Tree


| max_depth | val_rmse | test_rmse | ანალიზი                                 |
| --------- | -------- | --------- | --------------------------------------- |
| 3         | 0.2206   | 0.2368    | **Underfitting** — ხე ძალიან ზედაპირულია |
| 5         | 0.1948   | 0.1999    | უმჯობესდება                             |
| 10        | 0.1850   | 0.1900    | საუკეთესო                               |
| None      | 0.2079   | 0.2084    | **Overfitting** — სრულად გაზრდილი ხე    |


---

### Random Forest


| n_estimators | max_depth | val_rmse | test_rmse | ანალიზი                                  |
| ------------ | --------- | -------- | --------- | ---------------------------------------- |
| 10           | 3         | 0.2004   | 0.2169    | **Underfitting** — ცოტა ხე, მცირე სიღრმე |
| 100          | 10        | 0.1644   | 0.1725    | **საუკეთესო**                            |
| 300          | None      | 0.1673   | 0.1761    | მეტი ხე არ აუმჯობესებს       |


---

## Overfitting და Underfitting ანალიზი

### Overfitting მაგალითები

- **Linear Regression (165 სვეტი):** val_rmse=0.133, test_rmse=0.268 — 0.135 სხვაობა. მოდელმა დაიზეპირა train-ის პატერნები
- **Lasso alpha=0.0001 on Kaggle:** შიდა test-ზე 0.17, Kaggle-ზე 0.45 — linear მოდელი overfitting-ს განიცდის სრულიად ახალ მონაცემებზე და ამიტომაც მივიღე Kaggle-ზე იმაზე ბევრად ცუდი შედეგი, ვიდრე ტესტზე.

### Underfitting მაგალითები

- **Decision Tree depth=3:** val_rmse=0.220, test_rmse=0.237 — ხე ძალიან მარტივია, ვერ სწავლობს
- **Lasso alpha=0.1:** 48/50 სვეტი ნულდება — თითქმის არაფერი არ ისწავლა
- **Ridge alpha=500:** penalty ძალიან ძლიერია

### კარგი ბალანსი

- **Random Forest n=100, depth=10:** val_rmse=0.164, test_rmse=0.172

---

## საუკეთესო მოდელი

პირველად Lasso (alpha=0.0001) შეირჩა საუკეთესოდ შიდა test_rmse-ის მიხედვით (0.1702). თუმცა Kaggle-ზე მან 0.45 ქულა მიიღო.

Random Forest-მა (n=100, depth=10, leaf=2) შიდა test-ზე 0.1725 აჩვენა და Kaggle-ზე **0.17**. როგორც ჩანს ამ მოდელმა არ დააოვერფიტა.

**შერჩევის კრიტერიუმი:** ყველაზე დაბალი test_rmse სადაც `val_rmse ≈ test_rmse`.

---

## MLflow ექსპერიმენტები DagsHub-ზე

ყველა run დარეგისტრირებულია: [dagshub.com/sansi23/House-Prices---Advanced-Regression-Techniques](https://dagshub.com/sansi23/House-Prices---Advanced-Regression-Techniques)


| ექსპერიმენტი      | run-ების რაოდენობა |
| ----------------- | ------------------ |
| linear_regression | 3                  |
| ridge_regression  | 7                  |
| lasso_regression  | 7                  |
| decision_tree     | 11                 |
| random_forest     | 8                  |


თითოეულ run-ში დაილოგა:

- ყველა ჰიპერპარამეტრი
- Train / Val / Test მეტრიკები
- დატრენინგებული მოდელი 

საუკეთესო მოდელი (Random Forest) დარეგისტრირდა **Model Registry**-ში სახელით `house_prices_best_model`, version 2.

<img width="1185" height="68" alt="image" src="https://github.com/user-attachments/assets/2b0942b7-8887-4107-bea6-759b11bb1b4b" />

---

## გამოცდილება

- **შიდა test_rmse არ ნიშნავს იმას, რომ საუკეთესო მოდელია** — Lasso-მ შიდა test-ზე ყველაზე კარგი შედეგი დადო, მაგრამ Kaggle-ზე საკმაოდ ცუდი შედეგი აჩვენა.
- **Random Forest უფრო სტაბილურია** unseen data-ზე linear მოდელებთან შედარებით
- **ტრენინგისა და inference-ის ცალკე notebook-ებში შენახვა** workflow-ს სუფთად ინახავს
- **ყოველთვის train/test split** უნდა მოხდეს EDA-მდე

