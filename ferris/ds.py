class node:
    def __init__(self,data = None):
        self.data = data
        self.next = None
class llisty():
    def __init__(self):
        self.head = node()
    def append(self, item):
        cur = self.head
        new_node = node(item)
        while cur.next != None:
            cur = cur.next
        cur.next = new_node


    def length(self):
        cur = self.head
        n=0
        while cur:
            n+=1
            cur = cur.next
        return n

    def erase(self, index):
        if index >=self.length():
            print("Error: Out of range ")
            return
        cur_idx = 0
        cur =self.head
        while True:
            last_node =  cur
            cur = cur.next
            if cur_idx == index:
                last_node.next = cur.next
                return
            cur_idx += 1
    def insert_at(self, item, index):
        if index >= self.length():
            print('Error: Out of Range')
            return
        cur_idx = 0
        cur = self.head
        while True:
            new = node(item)
            last = cur
            cur = cur.next
            if cur_idx == index:
                last.next = new
                new.next = cur
                return
            cur_idx +=1
    def display(self):
        cur = self.head
        elems = []
        while cur.next:
            cur = cur.next
            elems.append(cur.data)
        print (elems)
mlist = llisty()
mlist.append(5)
mlist.append(7)
mlist.append(6)
mlist.append(2)
mlist.append(9)
mlist.display()
mlist.insert_at(1,4)
mlist.erase(5)
mlist.display()

