from sqlalchemy import create_engine, ForeignKey, String, Uuid, select, Table, Column
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
import uuid

class Base(DeclarativeBase):
    pass

book_genre = Table(
    'book_genres',
    Base.metadata,
    Column('book_id', Uuid, ForeignKey('books.id'), primary_key=True),
    Column('genre_id', Uuid, ForeignKey('genres.id'), primary_key=True)
)


class Genre(Base):
    __tablename__ = 'genres'
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(20))
    books: Mapped[list['Book']] = relationship(back_populates='genres', secondary='book_genres')


class Author(Base):
    __tablename__ = 'authors'
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(30))
    books: Mapped[list['Book']] = relationship(back_populates='author')
    

class Book(Base):
    __tablename__ = 'books'
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(20))
    id_author: Mapped[uuid.UUID] = mapped_column(ForeignKey('authors.id'))
    author: Mapped['Author'] = relationship(back_populates='books')
    genres: Mapped[list['Genre']] = relationship(back_populates='books', secondary='book_genres')
    

engine = create_engine("sqlite:///:memory:", echo=True)
Base.metadata.create_all(engine)

with Session(engine) as session:
    # Creación de autores
    author_1 = Author(name='Haruki Murakami')
    author_2 = Author(name='Ernest Hemingway')
    author_3 = Author(name='Patrick Rothfuss')

    # Creación de géneros (créalos UNA vez, y reusa los objetos en varios libros)
    g_drama = Genre(name='Drama')
    g_belico = Genre(name='Bélico')
    g_historico = Genre(name='Histórico')
    g_profundo = Genre(name='Profundo')
    g_fantastico = Genre(name='Fantástico')

    # Creación de libros — CADA libro puede llevar una LISTA de géneros
    book_1 = Book(name='For Whom The Bell Tolls', author=author_2, genres=[g_belico, g_historico])
    book_2 = Book(name='Tokyo Blues', author=author_1, genres=[g_drama])
    book_3 = Book(name='El nombre del viento', author=author_3, genres=[g_fantastico, g_drama])
    book_4 = Book(name='El temor de un hombre sabio', author=author_3, genres=[g_drama, g_fantastico])
    
    author_1.books.append(Book(name='Kafka en la orilla', genres=[g_drama, g_profundo]))
    author_2.books.append(Book(name='El viejo y el mar', genres=[g_profundo, g_drama]))

    session.add_all([author_1, author_2, author_3, book_1, book_2, book_3, book_4, g_drama, g_belico, g_historico, g_profundo, g_fantastico])
    session.commit()

    # Consulta de verificación — descomenta y ajusta esto al final
    print([g.name for g in book_1.genres])
    print([b.name for b in g_drama.books])
    

    