class node:
    def __init__(self,data=None, next = None):
        self.data = data
        self.next = next

class linked_list:
    def __init__(self):
        self.head = node()

    def append(self,data):
         new_node = node(data)
         cur = self.head
         while cur.next != None:
             cur= cur.next
         cur.next= new_node
    def length(self):
        cur = self.head
        total = 0
        while cur.next != None:
            total+=1
            cur = cur.next
        return total
    def display(self):
        elems=[]
        cur_node = self.head
        while cur_node.next:
            cur_node = cur_node.next
            elems.append(cur_node.data)
        print (elems)
    def get (self,index):
        if index>=self.length():
            print ("Error: 'Get' Index out of range!")
            return None
        cur_idx=0
        cur_node= self.head
        while True:
            cur_node = cur_node.next
            if cur_idx == index: return cur_node.data
            cur_idx += 1
    def erase(self,index):
        if index>=self.length():
            print ("Error: 'Erase' Index out of Range.")
            return
        cur_idx=0
        cur_node=self.head
        while True:
            last_node = cur_node
            cur_node = cur_node.next
            if cur_idx == index:
                lsst_node.next = cur_node.next
                return
            cur_idx+=1
    def prepend(self,data):
        new_node= node(data)
        new_node.next = self.head
        self.head= new_node

    def insert_after_node(self,prev_node, data):
        if not prev_node:
            print("Previous node is not in the list")
            return
        new_node= node(data)
        new_node.next = prev_node.next
        prev_node.next = new_node

    def set(self,index,data):
        if index>=self.length()or index<0:
            print("Error: 'Set' Index out of range!")
            return
        cur_node= self.head
        cur_idx= 0
        while True:
            cur_node= cur_node.next
            if cur_idx==index:
                cur_node.data=data
                return
            cur_idx+=1

    def reverse(self):
        cur = self.head
        prev= None
        while cur is not None:
            next = cur.next
            cur.next= prev
            prev = cur
            cur = next
        self.head = prev

    def reverserec(self, cur,prev):
        if cur.next is None:
            self.head = node
            cur.next = prev
            return
        next = cur.next
        cur.next = prev
        self.reverserec(next, cur)

    def reversere(self):
        if self.head is None:
            return
        self.reverserec(self.head, None)

    def printList(self):
        temp = self.head
        while (temp):
            print
            temp.data,
            temp = temp.next

my_list = linked_list()
my_list.append(1)
my_list.append(3)
my_list.append(5)
my_list.append(0)
my_list.append(8)
my_list.display()
print ("element at 3rd index:"+ str(my_list.get(3)))
my_list.set(0,34)
my_list.insert_after_node(my_list.head.next.next, 19)
my_list.display()

my_list.reversere()
my_list.printList()