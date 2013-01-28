import random

class PopMember:
      __slots__ = ('Weights', 'OnOff', 'Fitness')

#size of population, number of connections
def testPopulation(popsize,size):

      Population = makePopulation(popsize,size)
      
      Population = breedPop(Population, .1)
      
      Population = mutate(Population, .1)
      
      #printPop(Population)

def printPop(Pop):
      for i in range(0,len(Pop)):
            x = Pop[i]
            print(x.Weights)
            print(x.OnOff)
            print(x.Fitness)

def makePopulation(popsize,size):     

      Population = []
      
      for p in range(0, popsize):

            Weights = [1] * size
      
            OnOff = [1] * size

            for i in range(0,size):
                  w = random.uniform(0,1)
                  o = random.randrange(0, 2)

                  Weights[i] = w
                  OnOff[i] = o

            newMember = PopMember()
            newMember.Weights = Weights
            newMember.OnOff = OnOff
            newMember.Fitness = 0.0
            #newMember.Fitness = p
            
            
            Population.append(newMember)

      return Population
      
            
def sortPop(Pop):
      Pop.sort(key = lambda x: x.Fitness, reverse = True)
     
def breedPop(Pop, per):
      topNum = (int)(len(Pop) * per)
      #print(topNum)
      topNum = max(2, topNum)

      NewPop = []

      for i in range(0,topNum):
            NewPop.append(Pop[i])

      NewPop = makeChildren(NewPop, topNum, len(Pop))

      #print(len(NewPop))
      return NewPop

def makeChildren(Pop, top,sze):
      while len(Pop) < sze:
            #pick two parents
            parents = random.sample(range(top), 2)

            par1 = Pop[parents[0]]
            par2 = Pop[parents[1]]

            child = makeChild(par1, par2)

            Pop.append(child)
            
      return Pop


def makeChild(p1, p2):
      child = PopMember()

      xOvrW = random.sample(range(len(p1.Weights)), 1)[0]
      xOvrOO = random.sample(range(len(p1.OnOff)), 1)[0]

      newW = []

      newOO = []

      for i in range(0,len(p1.Weights)):
            if i < xOvrW:
                  newW.append(p1.Weights[i])
            else:
                  newW.append(p2.Weights[i])

            if i < xOvrOO:
                  newOO.append(p2.OnOff[i])
            else:
                  newOO.append(p2.OnOff[i])

      child.Weights = newW
      child.OnOff = newOO
      child.Fitness = 0

      return child

def mutate(pop, per):
      topNum = (int)(len(pop) * per)
      topNum = max(2, topNum)

      mutRate = .1
      mutAmount = .1
      
      for i in range(topNum, len(pop)):
            pop[i] = changeVals(pop[i], mutRate, mutAmount)

      return pop

def changeVals(member, mutRate, mutAmount):
      #mutate weights
      for i in range(0, len(member.Weights)):
            chance = random.random()
            if chance < mutRate:
                  rndAmt = random.uniform(-1.0, 1.0)
                  member.Weights[i] = member.Weights[i] * rndAmt * mutAmount
      #mutate OnOff
      for i in range(0, len(member.OnOff)):
            chance = random.random()
            if chance < mutRate:
                  rndChange = random.randrange(0, 2)
                  member.OnOff[i] = rndChange

      return member

testPopulation(3, 10)



