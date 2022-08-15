#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""Usage:

%holdem.py number_of_players"""

import functools
import itertools
import collections
import operator
import random

        
## Error when you try to deal from an empty deck    
class NoMoreCardsError(RuntimeError):
    """Y'all ran out of cards in the Deck"""
    
ACE = 14
KING = 13    
QUEEN = 12
JACK = 11
    
## Card Values: 2...
PIPS = collections.OrderedDict(
    [(2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'),
     (9, '9'), (10, 'T'), (JACK, 'J'), (QUEEN, 'Q'), (KING, 'K'), (ACE, 'A')])

## Invertion of PIPS dict (so you can make cards from characters).
IPIPS = {value: key for key, value in PIPS.iteritems()}

## Slang dictionary
SLANG = {value: key for key, value in
         dict(bullet=14, cowboy=13, lady=12, hook=11, dime=10, glock=9,
              snowman=8, hockey_stick=7, six=6, nickle=5, sailboat=4,
              trey=3, deuce=2).items()}
                                           

SPADE = 's'
CLUB = 'c'
HEART = 'h'
DIAMOND = 'd'

## Cart Suits (no ranking, ordering is for display only)
SUITS = collections.OrderedDict([('h', u"\u2661"),
                                 ('d', u"\u2662"),
                                 ('c', u"\u2663"),
                                 ('s', u"\u2660")])

## Reversed sorted
rsort = functools.partial(sorted, reverse=True)




## A Bag (counter) with method to invert and compute kick pips
class Ranker(collections.Counter):
    """extends collections.Counter (a multiset) which of course is
    perfect for defining both the mutliplicty and pips of poker
    hands.

    It adds 2 features:

    1) invert() method reversing key:value which becomes:
    multiplicity: card-pips, which is all that matters in evaluating
    nonflush/straight hands.

    2) tiebreak() method takes the inverted multiplicty and computes
    a unqie fraction that represent the strength of the hand vs
    hand of equal rank.
    """

    ## number of unqiue card pips
    base = len(PIPS)
    
    ## offset from 0
    bias = min(PIPS)
    
    ## Invert to {multiplicity: [card-pips]}
    def invert(self, D=functools.partial(collections.defaultdict, list)):
        result = D()
        for k, v in self.iteritems():
            result[v].append(k)
        return result

    ## Inversion
    __invert__ = invert
    
    ## Computer tie breaks and kickers
    def tiebreak(self):
        tmp = self.invert()
        keys = tmp.keys()
        k = []
        for key in rsort(keys):
            k.extend(rsort(tmp[key]))
        return self.kickers(*k)

    ## Kicker value of a card as an base-13 fraction, with bias.
    # \param v A card value \f$ v \in (2, \ellipsis, 14) \f$
    # \returns \f$ \frac{1}{13}(v-2) \in
    # (0, \ellipsis 0.9230769230769\ellipsis)\f$
    @classmethod
    def kicker(cls, v):
        from operator import truediv 
        return truediv(v - cls.bias, cls.base) 

    ## At 1st, 2nd, ... kickers in base 14 right side of decimal point
    # \param vs any number of kickers, ordered MSB first.
    # \returns kicker()'s as base 14 decimals.
    @classmethod
    def kickers(cls, *vs):
        return sum(cls.kicker(v) / (cls.base**count)  for count, v in
                   enumerate(vs))

    __float__ = tiebreak


    
## Some general methods for things with cards.
class _Base(object):

    def __repr__(self):
        return str(self)

    @property
    def show(self):
        print unicode(self)


## A Card
class Card(_Base):
    """Card(value, suit)"""

    _ascii_suits = dict(C='Clubs', D='Diamonds', H='Hearts', S='Spades')
    _full_names = {2: 'Deuce', 3: 'Trey', 4: 'Four', 5: 'Five', 6: 'Six',
                   7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten',
                   11: 'Jack', 12: 'Queen', 13: 'King', 14: 'Ace'}
    
    @classmethod
    def getall(cls, value):
        return [cls(value, suit=suit) for suit in PIPS]
    
    def __init__(self, value, suit=None):
        if suit is None:  # keyword (optional)
            suit = value[1].upper()
            value = IPIPS[value[0]]    
        assert value in PIPS, value
        assert suit in SUITS, suit
        self.value = value
        self.suit = suit

    def __int__(self):
        return self.value     
         
    def __cmp__(self, other):
        return cmp(self.value, other.value)

    def __eq__(self, other):
        return self.suit == other.suit and not cmp(self, other)
        
    def __str__(self):
        return "{}{}".format(self.pip, self.suit)
    
    def __unicode__(self):
        return unicode(PIPS[self.value]) +  SUITS[self.suit] + " "

    @property
    def pip(self):
        return PIPS[self.value]

    ## Check If argument in a card.
    @classmethod
    def iscard(cls, item):
        return isinstance(item, cls)
    
    def name(self, full=False):
        result = self._full_names[int(self)]
        if full:
            result += " of {}".format(self._ascii_suits[self.suit])
        return result

    def isface(self):
        return self.value in 'JQK'
        
## All 52 Cards
CARDS = list(itertools.starmap(Card, itertools.product(PIPS, SUITS)))



## Abstract Base Class for any grouping of cards (except a deck)
class Group(_Base):

    def __init__(self, *cards):
        assert len(cards) == self.number_of_cards, cards
        assert all(map(Card.iscard, cards)), cards
        #assert len(cards) == len(set(cards)), cards
        self.cards = cards

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

    def __iter__(self):
        return iter(self.cards)
        
    def __str__(self):
        return "".join(map(str, self))

    def __unicode__(self):
        return "".join(map(unicode, self))

    ## Deal from a ::Deck--a deck deosn't know about hands, but they've
    # heard of the deck.
    @classmethod
    def dealfrom(cls, deck, burn=True):
        if burn:  # keyword
            deck.pop()
        return cls(*map(apply, [deck.pop]*cls.number_of_cards))

    ## Sorted pips (high to low, Aces high).
    def pips(self):
        return tuple(rsort([card.value for card in self]))

    ## Suits 
    def suits(self):
        return set(card.suit for card in self)

    ## Sort the group
    def sort(self):
        return type(self)(*rsort(self))
        

## Hole cards:  2 in the hole
class Hole(Group):
    """hole = Hole(card, card)
    """
    
    class Category(collections.namedtuple("Category", "pips suited")):
        _suit = {False: 'o', True: 's'}
        def paired(self):
            return self.pips[0] == self.pips[1]
        
        def __repr__(self):
            result = "".join(map(PIPS.get, self.pips))
            if not self.paired():
                result += self._suit[self.suited]
            return result
    
    number_of_cards = 2

    @classmethod
    def all_deals(cls):
        return itertools.combinations(CARDS, 2)

    @classmethod
    def all_pairs(cls):
        for c1, c2 in cls.all_deals():
            if int(c1) > int(c2):
                continue
            yield cls(c1, c2)
    
    @classmethod
    def all_hands(cls, using=(HEART, CLUB)):
        first, second = using
        for hole in cls.all_pairs():
            c1, c2 = hole.cards
            if c1.suit != first or c2.suit not in using:
                continue
            yield hole

    ## Preflop Heads Up
    def __or__(self, other):
        assert type(self) is type(other)        
        deck = Deck()
        map(deck.remove, self.cards + other.cards)

        for i in range(0):
            deck.pop()
        
        wins = 0

        losses = 0

        for board in deck.all_boards():
            player = board.showdown(self, other)
            if player:
                losses += 1
            else:
                wins += 1
                
        return wins, losses

    def suited(self):
        return len(self.suits()) == 1

    def category(self):
        return self.Category(self.pips(), self.suited())

    ## Compute win/loss/tie for various card categories.
    @classmethod
    def compute_headsup(cls, n):
        wins = collections.defaultdict(int)
        losses = collections.defaultdict(int)
        ties = collections.defaultdict(int)

        counter = {-1: losses, 0: ties, 1: wins}

        try:
            for count in xrange(n):
                if count % 1000 == 0:
                    print count
                deck = Deck()
                h1, h2 = deck.deal_hole_cards(Hole.number_of_cards)
                board = River(*(deck.deal(Hand.number_of_cards)[0]))
                counter[cmp(board & h1, board & h2)][h1.category()] += 1
        except KeyboardInterrupt:
            pass
                
        #wins = {k: 100*v/float(n) for k, v in wins.items()}
        #losses = {k: 100*v/float(n) for k, v in losses.items()}
        #ties = {k: 100*v/float(n) for k, v in ties.items()}
        return wins, losses, ties
            
class Mixer(object):
        
    number_of_cards= 5

    def idraw(self, hole):
        deck = Deck()
        for card in deck:
            if card in hole or card in self:
                continue
            best = self.combine(list(hole) + [card])
            yield card, rsort(best)[0]
            
    def draw(self, hole):
        rank = 0
        best_card = [None]
        for card, best in self.idraw(hole):
            if float(best) < rank:
                continue
            if best.rank > rank:
                rank = best.rank
                best_card = [card]
            else: # best.rank == rank:
                best_card.append(card)
        return best_card, best
            
    ## Combine with Hole() cards and get a list of Hand() instances.
    def combine(self, hole):
        result = list(itertools.starmap(
            Hand,
            itertools.combinations(list(self) + list(hole),
                                   Hand.number_of_cards)))
        return result

    ## Combine with Hole() cards and 
    def top_ranks(self, hole):
        hands = self.combine(hole)
        bests = sorted(hands, key=float, reverse=True)
        return bests

    ## Board & Hole() --> best Hand().
    def __and__(self, hole):
        return sorted(self.combine(hole), reverse=True)[0]
        
    def __rand__(self, hole):
        return self & hole

    ## Compute best hand fromm hole cards
    # \param holes any number of Hole cards
    # \return int Winning player's parameter posisition (0, ..., N-1).
    def showdown(self, *holes):
        bests = [self & hole for hole in holes]
        ranks = map(float, bests)
        imax = ranks.index(max(ranks))
        return imax


    
## Community Cards ABC
class Community(Group, Mixer):

    ## Add a card to a concrete class and get the next concrete community
    # object
    def __add__(self, card):
        return self.next_(*(list(self) + [card]))

    ## Construc from a decl (after burning 1)
    def __lshift__(self, deck):
        deck.pop()
        return self + deck.pop()

    ## Run it n times
    # \param deck a Deck()
    # \param n NUmber of times to run it
    # \yields River() instances.
    def runit(self, deck, n):
        """for board in <community>.runit(deck, n):
        ...

        will run it n times from the deck.
        """
        for __ in xrange(n):
            # e.g.: river = (flop << deck) << deck
            yield reduce(operator.lshift,
                         [deck] * (Hand.number_of_cards-self.number_of_cards),
                         self)

    def allin(self, deck, n, *hands):
        for board in self.runit(deck, n):
            yield map(functools.partial(operator.and_, board), hands)
            

## 5th Street Board
class River(Group, Mixer):
    """River(*card) is best constructed from:

    >>>river = turn << deck

    >>>list = river.combine(hole)

    makes a list of possible hands, which can the be sorted with
    key=float.

    >>>best = river & hole

    will select the best
    """
    ## value of board
    def __float__(self):
        return float(Hand(*self))
    
    def nuts(self, *cards):
        print unicode(self)
        deck = Deck()
        filter(deck.remove, self)
        filter(deck.remove, cards)  # dead cards
        nuts = collections.defaultdict(list)
        for hole in itertools.starmap(Hole, itertools.combinations(deck, 2)):
            score = float(self & hole)
            nuts[score].append(hole)
        result = collections.OrderedDict()
        for score in sorted(nuts.keys()):
            result[score] = nuts[score]
        return result
        
    def plotnuts(self):
        from matplotlib import pylab as plt
        import numpy as np
        nuts = self.nuts()
        score = np.array(nuts.keys())
        count = np.array(map(len, nuts.values()))
        plt.plot(score, count, '-x')
        plt.xlim(0,9)
        plt.grid()
        plt.show()
        return nuts

    
## 4-th Street Board
class Turn(Community):
    """Turn(*flop, 4th) is made from:
    >>>turn = flop << deck
    """
    number_of_cards = 4
    next_ = River

    

## The 3 card Flop
class Flop(Community):
    """flop = Flop(deck.pop(), deck.pop(), deck.pop()) = deck.flop().

    Move onto the turn via:

    >>>turn = flop << deck
    """
    number_of_cards = 3
    next_ = Turn


            
    
HIGH = 'high card'
PAIR = 'pair'
PAIRS = 'two pairs'
TRIPS = '3 of a Kind'
STR8 = 'Sraight'
FLUSH = 'Flush'
BOAT = 'Full House'
QUADS = 'QUADS'
STR8FLUSH = 'STRAIGHT FLUSH'
ROYAL = """
RRR    OOO   Y    Y     A     L
R  R  O   O   Y  Y     A A    L
RRR   O   O    YY     AAAAA   L
R R   O   O    YY    A     A  L
R  R   OOO     YY   A       A LLLLLLL"""

handname = {0: HIGH,
            1: PAIR,
            2: PAIRS,
            3: TRIPS,
            4: STR8,
            5: FLUSH,
            6: BOAT,
            7: QUADS,
            8: STR8FLUSH,
            9: ROYAL}

STRAIGHTS = {'A5432': 5,
             '65432': 6,
             '76543': 7,
             '87654': 8,
             '98765': 9,
             'T9876': 10,
             'JT987': 11,
             'QJT98': 12,
             'KQJT9': 13,
             'AKQJT': 14}



## The classic 5-card Poker Hand
class Hand(Group):
    """hand = Hand(*cards)

    Ranking is as follows:

    0   High Card
    1   Pair
    2   Two Pair
    3   Three of a Kind
    4   Straight
    5   Flush
    6   Full House
    7   Four of a Kind
    8   Straight Flush
    9   Royal Flush

    sub-unit decimals are added to the rank integer to compare hands with the
    same class, and to break ties with kickers. The cards are arrange in order
    of poker-significane (e.g., in a boat it goes [trips-value, pair-value],
    while in 2-pair it goes [top pair value, bottom pair value, kicker]
    and then those numbers added in base-13 with decreasing significance
    from right to left.

    Hence, every hand can be ranked an compared to all other possible
    hands with a unique flow whose int classifies the hand, and whose
    fracitonal part determines relative value vs like hands.

    Player 1: 2♡2♤7♣J♢A♤    1.0588914954   A♤J♢

    """
    ## Five Card Hands
    number_of_cards= 5

    ## unranked on instaniation
    _rank = None

    ## Analyzer too
    _cpips = None
    
    ## Poker Sorter: Descending Multiplicity, Rank.
    # \param cards A list of cards
    # \return list cards sorted by degeneracy and magnitude.
    @staticmethod
    def _psort(cards):
        ref = map(int, cards).count
        # 2 step stable sort
        return rsort(rsort(cards), key=lambda y: ref(int(y)))
    
    def __init__(self, *cards):
        # sort card before making hand
        super(Hand, self).__init__(*self._psort(cards))
        
    ranks = {HIGH: 0, PAIR: 1, PAIRS: 2, TRIPS: 3, STR8: 4,
                 FLUSH: 5, BOAT: 6, QUADS: 7, STR8FLUSH: 8, ROYAL: 9}

    names = {value: key for key, value in ranks.items()}
    
    _rank_degeneracy = {(2, 1, 1, 1): PAIR,
                        (2, 2, 1): PAIRS,
                        (3, 1, 1): TRIPS,
                        (3, 2): BOAT,
                        (4, 1): QUADS}
        
    @property
    def rank(self):
        return self._rank or self._analyze()

    def degeneracy(self):
        return ~(self._cpips or Ranker(self.pips()))

    def _analyze(self):
        # count  pips.    
        self._cpips = Ranker(self.pips())
        # sort pips
        degeneracy = tuple(rsort(self._cpips.values()))
        # guess kicks
        kicks = float(self._cpips)
        try:
            key = self._rank_degeneracy[degeneracy]
        except KeyError:
            straight = self.straight()
            flush = self.flush()
            if straight:
                kicks = Ranker.kicker(straight)
                if flush:
                    if straight == ACE:
                        key = ROYAL
                        kicks = 0.
                    else:
                        key = STR8FLUSH
                else:
                    key = STR8
            else:
                if flush:
                    key = FLUSH
                else:
                    key = HIGH
        else:
            # compute kicks top down, starting with highest degerneracy   
            kicks = float(self._cpips)
        total =  self.ranks[key] + kicks
        self._rank = total    
        return total
        
    def __float__(self):
        return self._rank or self.rank

    def __complex__(self):
        return float(self) + 1j * self.nonstandard()

    def nonstandard(self, rerank={0:0, 1:1, 2:7, 3:11, 4:16, 5:22, 6:23,
                                  7:25, 8:26, 9: 28}):
        regular_rank = int(self)

        if regular_rank > 1:
            # return reranked regular hand, with kicks.
            return rerank[regular_rank] + (float(self) - int(self))

        ranker = self._cpips.invert()
        if len(ranker) == 1:
            # five of a kind
            return 29 + Ranker.kicker(ranker.values()[0])
        skeet = self.skeet()
        flush = self.flush()    
        if flush and skeet:
            # skeet flush
            return 27 + Ranker.kickers(*self.pips())
        
    def skeet(self):
        pips = self.pips()
        return (2 in pips and (3 in pips or 4 in pips) and 5 in pips and
                (6 in pips or 7 in pips or 8 in pips) and 9 in pips)

    def bobtail(self):
        for item in map(rsort, itertools.combinations(self.pips(), 4)):
            if self.makes_bobtails(item):
                return True
    
    @staticmethod
    def makes_bobtail(pips):
        pass
        
    def __int__(self):
        return int(float(self))
        
    def __cmp__(self, other):
        return cmp(self.rank, other.rank)

    ## Find Straight High Card
    # \returns int Highest pip in the straight (including wheel)
    def straight(self):
        """Highest card value in straight, or None"""
        return STRAIGHTS.get("".join(map(PIPS.get, self.pips())))
    
    def flush(self):
        """Boolean: IFF Flush"""
        return len(self.suits()) == 1


    
    
    
## Card Deck
class Deck(object):
    """deck = Deck()"""
    
    def __init__(self):
        cards = CARDS[::]
        self.cards = cards
        self.shuffle()
        
    def pop(self, index=0):
        try:
            return self.cards.pop(index)
        except IndexError as err:
            if index == 0:
                err = NoMoreCardsError
            raise err

                
    def __len__(self):
        return len(self.cards)    

    def shuffle(self):
        tmp = collections.deque(self.cards)    
        random.shuffle(tmp)
        self.cards = list(tmp)

    def __getitem__(self, index):   
        return self.cards[index]

    def __contains__(self, card):
        return card in self.cards

    def index(self, card):
        return self.cards.index(card)

    def remove(self, card):
        self.pop(self.index(card))

        
        
    def deal(self, m, n=1, klass=lambda *args: args):
        result = [[]]
        for item in xrange(n-1):
            result.append(list())
        result = tuple(result)
        for count in xrange(m):
            for player in xrange(n):
                result[player].append(self.pop())
        return list(itertools.starmap(klass, result))

    def deal_hole_cards(self, n):
        return self.deal(2, n=n, klass=Hole)

    def deal_stud5(self, n=1):
        return self.deal(5, n=n, klass=Hand)
        
    def flop(self):
        return Flop.dealfrom(self)

    def board(self):
        flop = self.flop()
        turn = flop << self
        board = turn << self
        return board

    def round(self, n):
        yield self.deal_hole_cards(n)
        yield self.board()
    

    ## Generate all possible boards from (remianing) deck
    # \yields River 
    def all_boards(self):
        """for river in deck.all_boards():
        ..."""
        for cards in itertools.combinations(self, 5):
            yield River(*cards)

    ## Generate __all__ hands
    # \param worst Min rank (inclusive)
    # \param best Max rank (exclusive)
    def all_hands(self, worst=0, best=10):
        """for hand in deck.all_hands(worst=0, best=10)..."""
        for cards in itertools.combinations(self, 5):
            hand = Hand(*cards)
            if worst <= hand.rank < best:
                yield hand

    def __unicode__(self):
        return u", ".join(map(unicode, self))
                
def round(n=2):
    D = Deck()
    holes = D.deal_hole_cards(n)
    for count, hole in enumerate(holes):
        print u'  Player {}: {}'.format(count+1, unicode(hole))#,
    print
    print '----------'
    flop = Flop.dealfrom(D)
    print 'Flop :', unicode(flop)
    turn = flop << D
    print 'Turn :', unicode(turn)
    board = turn << D
    print 'River:', unicode(board)
    print
    print
    winner = None
    for count, hole in enumerate(holes):
        best = board & hole
        print u'Player {}: {}    {}   {}  {} {}'.format(
            count + 1, unicode(best),
            best.rank,
            unicode(hole),
            str(best),
            handname[int(best.rank)])
        if winner is None or best > winner[0]:
            winner = (best, count+1, hole)


    print
    print 'Winner :', unicode(winner[0]), '  hole=', unicode(winner[-1]), handname[int(winner[0].rank)]
    #nuts =  list(board.nuts())[-1]
    #print float(nuts)
    return winner[0].rank
    
    
## A Round of Holdem
class Round(object):
    """round = Round('n_players')

    """
    
    board = None
    
    def __init__(self, n=2):
        ## A new DEck
        self.deck = Deck()
        ## dealt holes
        self.holes = self.deck.deal_hole_cards(n)
        ## Update Action
        self.action = u"  ".join(u"Player {}: {}".format(
            count+1, unicode(hole)) for count, hole in enumerate(self.holes))
        

    ## Flop
    def flop(self):
        self.board = self.deck.flop()
        self.action += u"\n-------------------------------\n Flop:  {}".format(
            unicode(self.board))
        return self.board

    ## Fourth and Fifth Street
    def street(self):
        try:
            self.board <<= self.deck
        except TypeError:
            raise RuntimeError("Dealing after the river?")
        self.action += u"\n         {}".format(unicode(self.board))        
        return self.board

    def showdown(self):
        bests = []
        for count, hole in enumerate(self.holes):
            best = self.board & hole
            self.action += u"\n Player {} {}     {}".format(count, unicode(best),
                                                            float(best))
            bests.append(best)
        return bests

    def __iter__(self):
        yield self.holes
        yield self.flop()
        yield self.street()
        yield self.street()
        for item in self.showdown():
            yield item
    
    def play(self):
        self.flop()
        self.street()
        self.street()
        return list(self.showdown())

    def __call__(self):
        return map(float, self.play())

    def __unicode__(self):
        return self.action

    ## Plot Head's Up Results
    @classmethod
    def plot(cls, n):
        from time import time
        from numpy import array
        from matplotlib import pylab as plt
        start = time()
        plt.figure(1)
        plt.xlim(0, 9)
        plt.ylim(0, 9)
        a, b = array(map(apply, map(cls, [2]*n))).T
        plt.plot(a, b, 'x')
        t = array(range(0, 10))
        plt.plot(t, t, 'r')
        plt.grid(True)
        plt.text(0.1, -0.8, 'High\nCard')
        plt.text(1.1, -0.5, 'Pair')
        plt.text(2.1, -0.8, 'Two\nPair')
        plt.text(3.1, -0.5, 'Trips')
        plt.text(4.1, -0.5, 'Straight')
        plt.text(5.1, -0.5, 'Flush')
        plt.text(6.1, -0.8, 'Full\nHouse')
        plt.text(7.1, -0.5, 'Quads')
        plt.text(8.1, -0.8, 'Straight\nFlush')
        return time() - start


        
def deal5():
    return Hand.dealfrom(Deck())
    

    
def heads_up_stats(n=1):
    while n:
        n -= 1
        yield Round(2)()


def gto(n, m):
    result = collections.Counter()        
    for showdown in headsup(n):
        result[tuple(showdown)] += 1
    return result
    
    
def histo(n):
    result = collections.Counter()
    while n:
        n -= 1
        result[Round(1)()[0]] += 1
    return result

def heads_up_scatter(n=1000):
    from numpy import array
    return array(map(apply, map(Round, [2]*n)))


def headsup(s1,s2,s3,s4):
    c1, c2, c3, c4 = map(Card, (s1, s2, s3, s4))    
    h1 = Hole(c1, c2)
    h2 = Hole(c3, c4)
    print u"{} vs {}".format(h1, h2)
    w, l = h1| h2
    print "{}    {}".format(w, l)
    return w, l
    

def deal_a_hand(hmin=0):
    deck = Deck()
    a, b = deck.deal_hole_cards(2)
    if min(map(int, b)) < hmin:
        return deal_a_hand(hmin=hmin)
    flop = deck.flop()
    turn = flop << deck
    river = turn << deck
    best = river & b
    return best

def count2(f, hmin=0):
    for c in itertools.count(1):
        h = deal_a_hand(hmin=hmin)
        if float(h) >= f:
            return c, h


def loop(n=0):
    rank = 0
    while rank < 7:
        rank = round(int(sys.argv[-1]))
        n+=1
    print n

if __name__ == '__main__':
    import sys
    #print sys.argv
    #np = sys.argv[-1]
    #print np
    round(int(sys.argv[-1]))
    #loop()
