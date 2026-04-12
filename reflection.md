# Reflection

- Working on this recommender showed how simple weights can feel intelligent and also mislead. When I boosted energy and lowered genre, the lists instantly became more diverse but also drifted from what the user asked for; flipping back proved how sensitive the system is to small tuning choices. 

- With only 10 songs, one track can jump several spots, so explainability really mattered—seeing “genre +2.0” and the exact energy gap made the tradeoffs obvious. 

- The biggest lesson is that data size and weight choices are both levers for bias, and you need clear explanations to notice when a “smart” result is actually off-target.
