---
layout: post
title:  "The value of information"
date:   2020-05-14 08:15:00 -0700
categories: maths
---

How much should you pay for information to help you make a decision?

Imagine that I run a business, and I'm thinking of launching a new product. I think it will sell well, but it's probably worth running a survey or doing some user research to be more certain. How much should I spend on that research?  How much should I _value_ it?

If I'm going to spend $100,000 developing and selling the product, that's the max I stand to lose, but if I'm already fairly certain it will succeed, wouldn't it be excessive to spend $100,000 on research?

This post builds upon ideas in the book [How To Measure Anything] by Douglas W. Hubbard, which I highly recommend. In the book, valuing information is a key part of a larger decision-making framework, but as I read it, I found myself wanting more mathematical justification for some of the statements and ideas.

This post fills in the gaps, by examining the question of how much one should pay for information using a simple gambling game under the lens of maths and probability. We'll start by figuring out how much you should pay for perfect information on the outcome of the game, then move on to imperfect information, which might be unreliable or even misleading. Finally, we'll come away with some rules of thumb for thinking about the value of information.


Jump to:

- [The dice game]
- [Perfect information]
- [Imperfect (misleading) information]
- [Imperfect (useless) information]
- [Rules of thumb]


## The dice game

I offer to play the following game as many times as you'd like. I'll roll a dice, and if it lands on 1 or 2, you lose $900. If it lands on 3, 4, 5 or 6, you win $60. Even though you're more likely to win than lose, the cost of losing is so great that your losses will far outstrip your winnings after a few games.

We can calculate the **[Expected Value]** of this game - how much you make or lose on average with each play - by multiplying the value of each outcome with its probability, and adding the results.

| Outcome     | Probability | Value   | Expected Value |
|-------------|-------------|---------|----------------|
| 1, 2        | 1/3         | -$900   | -\$300         |
| 3, 4, 5, 6  | 2/3         |  $60    | $40            |
| **Overall** |             |         | **-$260**      |

Or as totally legit math notation: first defining the [probability mass function] of a fair six-sided dice:

$$
\Pr(x)_X = \begin{cases} 1/6 &  x \in \{1, 2, 3, 4, 5, 6\} \\ 0 & \textrm{otherwise} \\ \end{cases}
$$

Then defining the game and computing its expected value as described above:

$$
\begin{align}
F(x) &= \begin{cases}-900 & x \in \{1, 2\}  \\ 60 & x \in \{3, 4, 5, 6\} \end{cases} \\
E[F(x)] &= \sum_{x \in \{1..6\}} \Pr(x)F(x) \\
&= \frac{1}{6} (60 + 60 + 60 + 60 - 900 - 900) \\
&= -260
\end{align}
$$

Given that you'll lose an average of $260 per game, you really shouldn't want to play it, so we'll call "don't play the game" your **default decision**. If you decide not to play, but I roll the dice to see what would have happened and it lands on a 5, you miss out on $60. That's **Opportunity Loss**. Since the chance of that happening is 2/3, the expected value of your opportunity loss, i.e. your **Expected Opportunity Loss**, is $40.

Here are the same definitions in handy table format:


| Term                      | Definition                                                                                             | Value in this game  |
|---------------------------|--------------------------------------------------------------------------------------------------------|---------------------|
| Default Decision          | The decision you would make given the information you have.                                            | Don't play!         |
| Opportunity Loss          | The amount that you expect to lose (or not make) if your decision was wrong                            | $60               |
| Expected Opportunity Loss (EOL) | The expected value of Opportunity Loss, i.e. the chance of each loss scenario multiplied by its value. | $60 * 2/3 = $40 |


## Perfect information

Now I'm going to give you the opportunity to learn the result of the game:

1. I'll roll the dice secretly,
2. For a price, I'll offer to tell you the result,
3. After buying (or not buying) the result, you'll decide whether to play.

Because I'm offering you _perfect information_ on the result of this game, you're never at risk of losing $900. You either:

- Don't buy the result, and don't play,
- Learn that you'll win and decide to play, or
- Learn that you'll lose, decide not to play.

In this situation, what's the absolute maximum you should pay to know the dice roll?

- Perhaps $60? No - if you pay $60, you'll make that money back 2/3 of the time, but lose it 1/3 of the time. Overall, you expect to lose $20 per game.
- Perhaps $0? That would be great for you, but I'm not gonna give you the information for free!

Let's solve this with maths. Defining $$k$$ as the amount paid for perfect information, we want to find the greatest value of $$k$$ such that the value of playing the game is at least $0. Adding in $$k$$, the game is defined as:

$$
G(x) = \begin{cases}- k + 0 & x \in \{1, 2\} \\ - k + 60 & x \in \{3, 4, 5, 6\} \end{cases}
$$

Solving $$E[G(x)] \geq 0$$ for $$k$$.

$$
\begin{align}
0 &\leq E[G(x)] \\
&\leq \sum_{x \in \{1..6\}} \Pr(x)G(x) \\
&\leq \frac{2}{6}(-k) + \frac{4}{6}(- k + 60)\\
&\leq -k + \frac{4 * 60}{6} \\
\therefore k &\leq40
\end{align}
$$

The absolute maximum you should pay is $40, at which point you expect to break even. If you can pay less, you'll make a profit. If you pay more, you'll lose in the long run. To demonstrate that, this table shows the expected value of each outcome when you pay $40 for information.

| Outcome     | Probability |          Value                        | Expected Value |
|-------------|-------------|:--------------------------------------|----------------|
| 1, 2        | 1/3         | -$40 + $0 = -$40 (decide not to play) | -$13.33        |
| 3, 4, 5, 6  | 2/3         | -$40 + $60 = $20 (decide to play)     |  $13.33        |
| **Overall** |             |                                       | **$0**         |



What we have found is that the **Expected Value of Perfect Information was equal to the Expected Opportunity Loss**, or: $$EVPI = EOL$$

Although perfect information is not always unattainable, it's clear that you'd  never want to pay more for _imperfect_ information than for perfect information. This has an interesting consequence: the _maximum_ you should be willing to pay for information is unrelated to the expected value of your default decision. If we changed the game so that you lose $9 million on a 1 or 2, it doesn't matter, since with perfect information you'll never risk that loss anyway!

On the other hand, if I occasionally lie to you, you definitely _would_ care if you stood to lose $9 million versus $900. The question is, how much would you care?

## Imperfect (misleading) information

In a new and dangerous version of the game, I offer you information for $$$k$$, but there's a chance $$p$$ that I'll lie to you. This will be represented by $$Y$$ (for "wh**y** would you lie to me?!").

$$
\Pr(y)_Y = \begin{cases} p &  lie \\ 1 - p & truth \\ \end{cases}
$$

The outcome of the game is determined by both the dice roll $$x$$, and whether I lied $$y$$. When I lie, you end up either losing $900, or missing out on $60.

$$
H(x, y) = \begin{cases}
- k + 0 & x \in \{1, 2\} \cap y = truth \\
- k -900 & x \in \{1, 2\} \cap y = lie \\
- k + 60 & x \in \{3, 4, 5, 6\} \cap y = truth \\
- k + 0 & x \in \{3, 4, 5, 6\} \cap y = lie \\
\end{cases}
$$

What is the maximum you should pay $$$k$$, in terms of the unreliability of information $$p$$? Solving $$E[H(x, y)] \geq 0$$ for $$k$$ and $$p$$:

$$
\begin{align}
0 &\leq E[H(x, y)] \\
&\leq \sum_{x \in \{1..6\},\\y \in \{truth, lie\}} \Pr(x)\Pr(y)H(x, y) \\
&\leq \frac{1}{3}(1-p)(-k) + \frac{1}{3}p(-k - 900) + \frac{2}{3}(1-p)(-k + 60) + \frac{2}{3}p(-k + 0)\\
\end{align}
$$

Putting that through the [magic math unicorn]:

$$
k \leq 40 - 340p \\
$$

To understand this, consider the extremes:

- If I never lie ($$p = 0$$), I'm giving you perfect information. We've already covered that case: $$k \leq \$40$$, where $40 is the Expected Opportunity Loss of not playing the game.
- If I always lie ($$p = 1$$), then I mislead you every time. I'd have to _give you_ $300 per game to make up for your losses. Note that $300 is the Expected Loss when playing the basic game!

Between $$p = 0$$ and $$1$$, the expected value of information varies linearly. If I lie to you less than $$p = 40/340 \approx 11.7\%$$ of the time, then there exists a price at which it's worth paying to play.


Let's take $$p=5\%$$ and $$k=0$$ as an example - so I give you the information for free but lie to you 1 in 20 times - and then compute the Expected Value of the game:

| Dice Roll   | I tell you... | Probability         | Value                         | Expected Value |
|-------------|---------------|---------------------|-------------------------------|----------------|
| 1, 2        | Lose (truth)  | 1/3 * 19/20 = 19/60 | $0 (decided not to play)      | $0             |
| 1, 2        | Win (lie)     | 1/3 * 1/20 = 1/60   | -$900 (mislead into playing)  | -$15           |
| 3, 4, 5, 6  | Win (truth)   | 2/3 * 19/20 = 38/60 | $60 (decided to play)         | $38            |
| 3, 4, 5, 6  | Lose (lie)    | 2/3 * 1/20 = 2/60   | $0 (mislead into not playing) | $0             |
| **Overall** |               |                     | -                             | **$23**        |

The expected value of the game - $23 - is exactly what the math above gave us for the maximum we'd want to pay for information: $$k = 40 - 340/20 = $23$$.

There is another interesting way of looking at this: the Expected Opportunity Loss of the game with 95% truthful information is:

- $15 - if you are mislead into playing on a 1 or 2
- plus $2 - if you are mislead into not playing on a 3, 4, 5, 6
- equals $17.

With the original strategy (don't play), you stand to lose out on $40 if you're wrong. With the new strategy (play if I tell you you'll win), you only risk losing or missing out on $17. Thus, Expected Loss has been reduced by $40 - $17, which is $23!

And _that_ is how How To Measure Anything frames the Expected Value of Information: $$EVI = EOL_{before} - EOL_{after}$$. That's why the $$EVPI = EOL$$, because perfect information reduces the chance of a bad decision to zero.


## Imperfect (useless) information

Finally, we can also imagine a version where instead of truth or lies, I give you truth or nothing. When you pay for information but receive nothing, you wasted your money and you just have to stick with your default decision to not play. If there's even a small chance I might give you the truth then there should be _some price_ worth paying, but if I'll never tell you anything, then the information is clearly worth nothing. Symbolically, $$k \leq 40 - 40p$$.

This suggests two categories of imperfect information: potentially useless information, and potentially misleading information.

**Potentially useless information** might influence you to move from your default decision, but only if it was wrong. For example: an office manager of a small company is deciding whether to provide a vegetarian option for lunch. If the number of vegetarians exceeds a threshold they'll include the option, but since they've not heard that anyone is vegetarian they might default to not doing so. In that situation, a survey of the office can only help, but it could be useless if too many vegetarians miss the survey.

**Potentially misleading information** might influence you to move from your default decision, even if it was _right_. Taking the example from the opening of product research, perhaps I'm so confident in the product that my default decision is to go ahead with it. If I run a survey of potential customers, and it's answered by an unrepresentative slice of the population who all hate it, it could persuade me to abandon the project even if it was a good idea.

# Rules of thumb for thinking about the value of information

In summary, the situations examined in this post suggest two guiding principles for valuing information.

Firstly, **the most you should pay for information is the amount you expect to lose if your default decision is wrong**. This is because the most you'll pay for information is whatever you would pay for perfect information, and perfect information cannot cause you to change to a bad decision.

Secondly, **strive to collect information which is not _potentially misleading_**. When the worst case is that information is useless, you don't have to consider the cost of switching to a bad decision, but when information can mislead you, then you must take into account the cost of being persuaded to make a bad decision.

It's worth noting that these principles rely on your initial estimates being vaguely correct, or at least honestly representative of how little you know. Producing [calibrated estimates] is a skill you can learn (another topic covered by Hubbard in How To Measure Anything).

Ultimately, these were just simple examples. In the real world, there will usually be many variables at play, and decisions are often on a continuum (not just yes/no), so I hope to cover more complex situations in a future post. I wrote this post so that I could feel that I truly understand and believe Hubbard's statements on the value of information - with worked examples to reference - and in that sense it has been a success for me. I hope there was something interesting in here for you, too!


[probability mass function]: https://en.wikipedia.org/wiki/Probability_mass_function#Finite
[magic math unicorn]: https://www.wolframalpha.com/input/?i=%5Cfrac%7B1%7D%7B3%7D%281-p%29%28-k%29+%2B+%5Cfrac%7B1%7D%7B3%7Dp%28-k+-+900%29+%2B+%5Cfrac%7B2%7D%7B3%7D%281-p%29%28-k+%2B+60%29+%2B+%5Cfrac%7B2%7D%7B3%7Dp%28-k+%2B+0%29+%3E%3D+0
[Expected Value]: https://en.wikipedia.org/wiki/Expected_value#Finite_case
[How To Measure Anything]: https://www.howtomeasureanything.com/3rd-edition/
[calibrated estimates]: https://en.wikipedia.org/wiki/Calibrated_probability_assessment

[The dice game]: #the-dice-game
[Perfect information]: #perfect-information
[Imperfect (misleading) information]: #imperfect-misleading-information
[Imperfect (useless) information]: #imperfect-useless-information
[Rules of thumb]: #rules-of-thumb-for-thinking-about-the-value-of-information
