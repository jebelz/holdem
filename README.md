# holdem
Texas Holdem Script
This file runs as a shell script. It deals a hand of Texas Hold 'Em and ranks the players hands (including card values of a made hand,
along with kickers.

It's not a game; it's more base classes for card games in general.


>./holdem.py 4
  Player 1: 5♣ 5♢ 
  Player 2: 8♠ J♠ 
  Player 3: 3♢ K♣ 
  Player 4: 4♠ J♡ 

----------
Flop : 4♣ 6♣ A♢ 
Turn : 4♣ 6♣ A♢ 2♡ 
River: 4♣ 6♣ A♢ 2♡ J♣ 


Player 1: 5♣ 5♢ A♢ J♣ 6♣     1.30601169427   5♣ 5♢   5c5dAdJc6c pair
Player 2: J♣ J♠ A♢ 8♠ 6♣     1.7661846574   8♠ J♠   JcJsAd8s6c pair
Player 3: A♢ K♣ J♣ 6♣ 4♣     0.992407613394   3♢ K♣   AdKcJc6c4c high card
Player 4: J♣ J♡ 4♣ 4♠ A♢     2.70960400546   4♠ J♡   JcJh4c4sAd two pairs

Winner : J♣ J♡ 4♣ 4♠ A♢    hole= 4♠ J♡  two pairs
