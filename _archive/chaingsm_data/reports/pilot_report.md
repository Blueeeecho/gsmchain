# Pilot Report

## Summary

- selected base samples: chaingsm_train_000001, chaingsm_train_000002
- total records: 10
- generated variants: 8
- failed variants: 0
- validator pass rate: 8/8

## Base Sample chaingsm_train_000001

### Original Question

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

### Original Answer

Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

### Original Final Answer

72

### Variant: independent_decoy

Generated question:

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Additionally, her friend Maria sold 30 clips in April and twice that many in May. How many clips did Natalia sell altogether in April and May?

Core chain:

```json
[
  [
    "Natalia_april",
    "Natalia_may",
    "/2"
  ],
  [
    "Natalia_april",
    "Natalia_total",
    "aggregate_sum"
  ],
  [
    "Natalia_may",
    "Natalia_total",
    "aggregate_sum"
  ]
]
```

Distractor chain:

```json
[
  [
    "Maria_april",
    "Maria_may",
    "*2"
  ],
  [
    "Maria_april",
    "Maria_total",
    "aggregate_sum"
  ],
  [
    "Maria_may",
    "Maria_total",
    "aggregate_sum"
  ]
]
```

Gold expression: 48 + 48/2

Distractor expression: 30 + 30*2

Difficulty tags:

```json
{
  "entity_overlap": "low",
  "operation_similarity": "high",
  "answer_proximity": "near",
  "computational_complexity": "simple"
}
```

### Variant: attribute_mismatch

Generated question:

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Each clip was sold for $3. How many clips did Natalia sell altogether in April and May?

Core chain:

```json
[
  [
    "Natalia_sold_April",
    "Natalia_sold_May",
    "/2"
  ],
  [
    "Natalia_sold_April",
    "total_clips",
    "+ Natalia_sold_May"
  ]
]
```

Distractor chain:

```json
[
  [
    "price_per_clip",
    "April_revenue",
    "*48"
  ],
  [
    "Natalia_sold_May",
    "May_revenue",
    "*3"
  ],
  [
    "April_revenue",
    "total_revenue",
    "+ May_revenue"
  ]
]
```

Gold expression: 48 + (48/2)

Distractor expression: 48*3 + (48/2)*3

Difficulty tags:

```json
{
  "entity_overlap": "high",
  "operation_similarity": "high",
  "answer_proximity": "far",
  "computational_complexity": "multi_step"
}
```

### Variant: path_competition

Generated question:

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. She sold the same number of clips to each friend, so the number of friends she sold to in May is also half of April's friends. How many clips did Natalia sell altogether in April and May?

Core chain:

```json
[
  [
    "April_clips",
    "May_clips",
    "/2"
  ],
  [
    "April_clips",
    "Total_clips",
    "+"
  ]
]
```

Distractor chain:

```json
[
  [
    "April_friends",
    "May_friends",
    "/2"
  ],
  [
    "April_friends",
    "Total_friends",
    "+"
  ]
]
```

Gold expression: 48 + 48/2

Distractor expression: 48 + 48/2

Difficulty tags:

```json
{
  "entity_overlap": "high",
  "operation_similarity": "high",
  "answer_proximity": "same",
  "computational_complexity": "simple"
}
```

### Variant: target_scope_misalignment

Generated question:

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Then in June, she sold three times as many clips as she sold in April. How many clips did Natalia sell altogether in April and May?

Core chain:

```json
[
  [
    "clips_April",
    "clips_May",
    "/2"
  ],
  [
    "clips_April",
    "total_AM",
    "+clips_May"
  ]
]
```

Distractor chain:

```json
[
  [
    "clips_April",
    "clips_June",
    "*3"
  ]
]
```

Gold expression: 48 + (48 / 2)

Distractor expression: 48 * 3

Difficulty tags:

```json
{
  "entity_overlap": "high",
  "operation_similarity": "medium",
  "answer_proximity": "far",
  "computational_complexity": "simple"
}
```

## Base Sample chaingsm_train_000002

### Original Question

Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

### Original Answer

Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

### Original Final Answer

10

### Variant: independent_decoy

Generated question:

Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. Also, Ben earns $10 an hour for gardening and spent 45 minutes gardening. How much did Weng earn?

Core chain:

```json
[
  [
    "Weng_rate_per_hour",
    "Weng_rate_per_minute",
    "/60"
  ],
  [
    "Weng_rate_per_minute",
    "Weng_earnings",
    "*50"
  ]
]
```

Distractor chain:

```json
[
  [
    "Ben_rate_per_hour",
    "Ben_rate_per_minute",
    "/60"
  ],
  [
    "Ben_rate_per_minute",
    "Ben_earnings",
    "*45"
  ]
]
```

Gold expression: 12/60*50

Distractor expression: 10/60*45

Difficulty tags:

```json
{
  "entity_overlap": "low",
  "operation_similarity": "high",
  "answer_proximity": "far",
  "computational_complexity": "simple"
}
```

### Variant: attribute_mismatch

Generated question:

Weng earns $12 an hour for babysitting and $8 an hour for gardening. Yesterday, she did 50 minutes of babysitting and 1 hour of gardening. How much did she earn from babysitting?

Core chain:

```json
[
  [
    "Weng_babysitting_hourly",
    "per_minute_wage",
    "/60"
  ],
  [
    "per_minute_wage",
    "babysitting_earnings",
    "*50"
  ]
]
```

Distractor chain:

```json
[
  [
    "Weng_gardening_hourly",
    "gardening_earnings",
    "*1"
  ]
]
```

Gold expression: 12/60*50

Distractor expression: 8*1

Difficulty tags:

```json
{
  "entity_overlap": "high",
  "operation_similarity": "low",
  "answer_proximity": "far",
  "computational_complexity": "simple"
}
```

### Variant: path_competition

Generated question:

Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. Her friend Li earns $15 per hour and worked for 1 hour. How much did Weng earn?

Core chain:

```json
[
  [
    "Weng_hourly_rate",
    "Weng_minute_rate",
    "/60"
  ],
  [
    "Weng_minute_rate",
    "Weng_earnings",
    "*50"
  ]
]
```

Distractor chain:

```json
[
  [
    "Li_hourly_rate",
    "Li_earnings",
    "*1"
  ]
]
```

Gold expression: 12/60*50

Distractor expression: 15*1

Difficulty tags:

```json
{
  "entity_overlap": "medium",
  "operation_similarity": "high",
  "answer_proximity": "far",
  "computational_complexity": "simple"
}
```

### Variant: target_scope_misalignment

Generated question:

Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. She plans to babysit for 2 hours tomorrow. How much did she earn?

Core chain:

```json
[
  [
    "12",
    "0.2",
    "/60"
  ],
  [
    "0.2",
    "10",
    "*50"
  ]
]
```

Distractor chain:

```json
[
  [
    "12",
    "24",
    "*2"
  ]
]
```

Gold expression: 12/60*50

Distractor expression: 12*2

Difficulty tags:

```json
{
  "entity_overlap": "medium",
  "operation_similarity": "low",
  "answer_proximity": "far",
  "computational_complexity": "multi_step"
}
```
