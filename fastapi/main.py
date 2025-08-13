from fastapi import FastAPI
import random
from pydantic import BaseModel
from typing import Optional,Union
import sqlite3


class Product(BaseModel):
    name:str
    price:float
    stock:int
    category:Union[str,None] = None
    inStock : Optional[bool] = None


def init_db():
    conn = sqlite3.connect('veritabani.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            price REAL,
            stock INTEGER,
            category TEXT,
            inStock BOOLEAN
        )
    """)    
    conn.commit()
    conn.close()


init_db()
app = FastAPI()

@app.get("/")
def read_root():
    return {"output":"umutsuz durum yoktur umutsuz insanlar vardir.","durum":True}


@app.get("/soz/{person}")
def soz(person:str):
    words = {
        "ataturk": "Umutsuz durumlar yoktur, umutsuz insanlar vardir",
        "mevlana": "Ne olursan ol yine gel",
        "tesla":"Hayal gücü her şeydir, Hayal gücü geleceğin önizlemesidir"
    }
    if person=="rastgele":
        output = random.choice(list(words.values()))
    else:
        output=words.get(person.lower())
    return {"output":output}


@app.post("/urun/")
def product(product:Product):
    product_dict = product.dict()
    product.stock = max(0,product.stock)
    product_dict.update({"stock":product.stock})
    product_dict.update({"inStock":product.stock>0})
    return {"output":product_dict}


@app.post("/urun/ekle")
def add_product(product: Product):
    conn = sqlite3.connect("veritabani.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO products(name,price,stock,category,inStock) VALUES(?,?,?,?,?)",
        (product.name, product.price, product.stock, product.category, product.inStock)
    )    
    conn.commit()
    conn.close()
    return {"output":"Ürün başarıyla eklendi."}
