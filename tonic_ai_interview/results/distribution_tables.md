# Distribution study - generated tables (distribution_study.py, seed 0)

### Doc level

```
                     train       val  vendor_a  vendor_b
docs               372.000    40.000   100.000    30.000
transcripts        186.000    20.000   100.000     0.000
notes              186.000    20.000     0.000    30.000
median_chars      3531.000  3310.000  4476.000  2746.000
max_chars         7272.000  5211.000  6701.000  3918.000
median_words       514.000   533.000   772.000   382.000
%docs_markdown      50.000    50.000     0.000   100.000
%docs_bullets       50.000    50.000     0.000   100.000
fillers/1k_words     3.200     4.900    13.900     0.000
lowercase_share      0.914     0.922     0.908     0.936
digits/1k_words     68.900    70.400     0.000   122.400
```

### Speaker tags (count of 'ROLE:' line starts)

```
                             train  val  vendor_a  vendor_b
PHYSICIAN                      612   48       315         0
PATIENT                       2498  214      1298         0
NURSE                          767   81       338         0
CARE_COORDINATOR               592   67       457         0
FAMILY_CAREGIVER                48    9        55         0
ADVANCED_PRACTICE_CLINICIAN    601   23       216         0
```

### Overlap / id collisions with train+val

```
                             vendor_a  vendor_b
exact text dup of train/val         0         0
id also used in val                 0         0
```

### Model-tokenizer length

```
                    train     val  vendor_a  vendor_b
median_tokens       870.0   833.0    1109.0     669.0
max_tokens         1807.0  1343.0    1652.0     942.0
%docs_>512_tokens    98.9   100.0     100.0      96.7
```

### Gold span counts

```
set                train  val  vendor_a  vendor_b
label                                            
name                1203  125        64       169
drug_name           1881  235       631       136
drug_amount         1099  150       356        86
medical_condition   2139  271       481       275
date                2347  266         0        80
```

### Gold spans per doc

```
set                train   val  vendor_a  vendor_b
label                                             
name                3.23  3.12      0.64      5.63
drug_name           5.06  5.88      6.31      4.53
drug_amount         2.95  3.75      3.56      2.87
medical_condition   5.75  6.78      4.81      9.17
date                6.31  6.65      0.00      2.67
```

### Label share (% of spans)

```
set                train   val  vendor_a  vendor_b
label                                             
name                13.9  11.9       4.2      22.7
drug_name           21.7  22.4      41.2      18.2
drug_amount         12.7  14.3      23.2      11.5
medical_condition   24.7  25.9      31.4      36.9
date                27.1  25.4       0.0      10.7
```

### Mean span length (words)

```
set                train   val  vendor_a  vendor_b
label                                             
name                1.83  1.85      1.00      2.04
drug_name           1.16  1.20      1.08      1.43
drug_amount         1.99  2.01      2.54      1.97
medical_condition   2.05  2.00      2.03      1.88
date                2.49  2.33       NaN      2.38
```

### Gold spans per doc, same doc type

```
set                train_transcripts  vendor_a  train_notes  vendor_b
label                                                                
name                            0.72      0.64         5.75      5.63
drug_name                       5.88      6.31         4.24      4.53
drug_amount                     3.57      3.56         2.34      2.87
medical_condition               4.44      4.81         7.06      9.17
date                            7.98      0.00         4.64      2.67
```

### date surface form, same doc type (% of date spans)

```
set            train_notes  train_transcripts  vendor_b
form                                                   
MM/DD/YYYY             8.0                0.0      23.8
Month D, YYYY         47.3                0.0      68.8
Month YYYY             0.2                0.0       0.0
month only             0.0                0.5       0.0
other                  1.6                5.3       0.0
relative              42.1               88.2       0.0
weekday                0.0                5.9       0.0
year only              0.8                0.1       7.5
```

### name flags, same doc type (% of name spans)

```
set                              train_notes  train_transcripts  vendor_a  vendor_b
title inside span (Dr./Mr./Ms.)          0.0                0.0       0.0       0.0
single token                            25.4              100.0     100.0      23.1
all lowercase                            0.0                0.0      43.8       0.0
ALL CAPS                                 0.0                0.0       0.0       0.0
apostrophe or hyphen                     0.6                0.0       1.6       5.9
```

### date: surface form (% of date spans)

```
set             train    val  vendor_b
form                                  
MM/DD/YYYY        2.9    5.3      23.8
Month D, YYYY    17.4   15.0      68.8
Month YYYY        0.1    0.0       0.0
month only        0.3    0.4       0.0
other             4.0    3.4       0.0
relative         71.2   72.6       0.0
weekday           3.7    2.6       0.0
year only         0.3    0.8       7.5
(n spans)      2347.0  266.0      80.0
```

### drug_amount: number format (% of spans)

```
set                           train    val  vendor_a  vendor_b
form                                                          
bare number                     0.1    0.0       0.0       0.0
digit+unit, no space (10mg)     1.4    1.3       0.0       3.5
digit+unit, space (10 mg)      98.4   95.3       0.0      96.5
other                           0.2    0.0       2.5       0.0
spelled-out number              0.0    3.3      97.5       0.0
(n spans)                    1099.0  150.0     356.0      86.0
```

### drug_amount: flags (% of spans)

```
set                             train    val  vendor_a  vendor_b
unit mg / milligram              76.6   64.7      85.4      39.5
unit mcg/microgram                1.3    2.7       3.4       0.0
unit units/UNT                   12.4   20.0       3.7      36.0
unit ml                          15.4   24.0       0.0      52.3
unit tablet/pill/puff/capsule     2.1    2.0       5.3       0.0
frequency inside span             0.0    0.0       0.0       0.0
(n spans)                      1099.0  150.0     356.0      86.0
```

### name: flags (% of spans)

```
set                               train    val  vendor_a  vendor_b
title inside span (Dr./Mr./Ms.)     0.0    0.0       0.0       0.0
single token                       33.7   36.8     100.0      23.1
all lowercase                       0.0    0.0      43.8       0.0
ALL CAPS                            0.0    0.0       0.0       0.0
apostrophe or hyphen                0.5    7.2       1.6       5.9
(n spans)                        1203.0  125.0      64.0     169.0
```

### drug_name: flags (% of spans)

```
set                        train    val  vendor_a  vendor_b
Capitalised first letter    36.2   43.4       0.2      92.6
all lowercase               63.5   56.6      99.8       7.4
multi-word                  15.7   19.6       7.8      41.9
hyphen combo (a-b)           5.5    0.4       2.4       0.0
brackets [Brand]             0.0    0.0       0.0       0.0
(n spans)                 1881.0  235.0     631.0     136.0
```

### medical_condition: flags (% of spans)

```
set                        train    val  vendor_a  vendor_b
abbreviation (2+ caps)       9.6   12.5       0.2      24.4
Capitalised first letter    33.6   38.0       0.8      59.3
all lowercase               66.2   60.9      99.2      40.4
contains digit/stage        14.4   19.6       4.2      27.3
spelled-out number           0.0    0.0       8.1       0.0
(n spans)                 2139.0  271.0     481.0     275.0
```

### OOV vs train gold spans of same label (lowercased, verbatim)

```
                                n  OOV% occurrences  OOV% distinct
name              val       125.0              77.6           81.8
drug_name         val       235.0               0.9            2.0
drug_amount       val       150.0               6.7           20.5
medical_condition val       271.0               5.2           12.9
date              val       266.0              15.8           32.0
name              vendor_a   64.0              95.3           92.9
drug_name         vendor_a  631.0              20.4           28.6
drug_amount       vendor_a  356.0             100.0          100.0
medical_condition vendor_a  481.0              10.2           19.4
date              vendor_a    0.0               NaN            NaN
name              vendor_b  169.0              76.3           78.0
drug_name         vendor_b  136.0               1.5            8.0
drug_amount       vendor_b   86.0               2.3           14.3
medical_condition vendor_b  275.0               2.9           14.8
date              vendor_b   80.0              36.2           52.7
```

### Random gold spans, label=name (seed 0)

```
           train                   val vendor_a            vendor_b
0   Letha Hirthe  Karry Shandra Crooks  dumitru        Emily Parker
1   Michael Tran                  Ryan   Dennis             Bradtke
2         Horace        Emily Mitchell     Jill  Carlo Von Gislason
3  Jane Peterson               Brayden    tommy  Classie Lulu Klein
4    Michael Lin                Russel    Tommy       Emily Roberts
5          Loree               Cynthia    rowan               Carlo
```

### Random gold spans, label=drug_name (seed 0)

```
                  train            val            vendor_a      vendor_b
0           hydralazine   Epoetin Alfa       amitriptyline   Simvastatin
1               aspirin      Torsemide  metoprol succinate    Lisinopril
2            prednisone     sertraline         doxycycline     Metformin
3             ibuprofen  Acetaminophen           glipizide  Epoetin Alfa
4  sacubitril-valsartan      nebivolol           albuteral   Simvastatin
5                  DMPA     amlodipine       canagliflozin    Furosemide
```

### Random gold spans, label=drug_amount (seed 0)

```
      train            val                  vendor_a      vendor_b
0     25 mg          10 mg            ten milligrams    4000 units
1  16 units        0.25 mg                 ten units          1 mL
2    500 mg          10 mg          fifty milligrams         10 mg
3      5 mg  18 micrograms            ten milligrams         10 mg
4    300 mg          30 mg  three hundred milligrams        500 mg
5     50 mg          20 mg          forty milligrams  0.0272 MG/MG
```

### Random gold spans, label=medical_condition (seed 0)

```
                 train                           val                            vendor_a                        vendor_b
0             shingles                      diabetes     gastroesophageal reflux disease                            ESRD
1  atrial fibrillation                        Anemia  chronic kidney disease stage three  Chronic Kidney Disease Stage 4
2   Seasonal Allergies                        asthma                              stroke                        cavities
3                 ESRD           atrial fibrillation                        hypertension                            ESRD
4        kidney stones       End-Stage Renal Disease                              anemia                 Acute sinusitis
5             shingles  generalized anxiety disorder                        hypertension          ischemic heart disease
```

### Random gold spans, label=date (seed 0)

```
              train                val vendor_a           vendor_b
0         yesterday              today                        1985
1       in 3 months          next week               March 5, 1951
2  October 18, 2023        a month ago           November 15, 2023
3             today    Tuesday morning            January 15, 1957
4       in 3 months     three days ago            October 15, 2023
5             today  the last few days           November 15, 2023
```

### Entity-looking text left unlabeled

```
                                                             matches not in any gold span % unlabeled                                       example
date: MM/DD/YYYY                                    train         70                    0         0.0                                              
                                                    val           14                    0         0.0                                              
                                                    vendor_a       0                    0        None                                              
                                                    vendor_b      19                    0         0.0                                              
date: Month D, YYYY                                 train        413                    0         0.0                                              
                                                    val           40                    0         0.0                                              
                                                    vendor_a       0                    0        None                                              
                                                    vendor_b      55                    0         0.0                                              
date: relative (yesterday/last week/ago/in N weeks) train        799                    7         0.9                      trn-n-016: 'in 6 months'
                                                    val           78                    2         2.6                      dev-n-018: 'in 3 months'
                                                    vendor_a     357                  356        99.7                           va-001: 'yesterday'
                                                    vendor_b      34                   34       100.0                    dev-n-001: 'in four weeks'
amount: digits + mg                                 train        867                   25         2.9                            trn-n-134: '42 mg'
                                                    val           98                    6         6.1                           dev-n-005: '3.5 mg'
                                                    vendor_a       0                    0        None                                              
                                                    vendor_b      36                    2         5.6                            dev-n-029: '58 mg'
amount: spelled + milligrams                        train          0                    0        None                                              
                                                    val            2                    0         0.0                                              
                                                    vendor_a     310                    8         2.6  va-069: 'one hundred thirty four milligrams'
                                                    vendor_b       0                    0        None                                              
name: Dr. Surname                                   train        441                    0         0.0                                              
                                                    val           42                    0         0.0                                              
                                                    vendor_a       0                    0        None                                              
                                                    vendor_b      70                    0         0.0                                              
```
