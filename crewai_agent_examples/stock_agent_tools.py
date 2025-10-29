from crewai.tools import BaseTool, tool
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Float
import logging
from datetime import datetime
from pydantic import BaseModel, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Database setup for caching
Base = declarative_base()


class StockAnalysisData(BaseModel):
    stock_name: str
    stock_code: str
    market: str
    buy_price: float
    target_price_daily: float
    target_price_weekly: float
    stop_loss: float
    analysis_date_time: datetime
    analysis: str

    @validator("analysis_date_time", pre=True, always=True)
    def parse_analysis_date_time(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value


class StockMarketAnalysisData(Base):
    """Database model for SharePoint file caching."""
    __tablename__ = 'stock_market_data_analysis'

    stock_name = Column(String(32), primary_key=True)  # MD5 hash as primary key
    stock_code = Column(String(20), nullable=False)
    market = Column(String(20), nullable=False, index=True)
    buy_price = Column(Float, nullable=False)
    target_price_daily = Column(Float, nullable=False)
    target_price_weekly = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False, default=0)
    analysis_date = Column(DateTime, nullable=False, default=datetime.utcnow())
    day_end_price = Column(Float, nullable=True)

class DatabaseClient:
    def __init__(self):
        # Initialize database connection here
        self.connection_string=f"postgresql+psycopg2://dev_user:dev_password@localhost:5432/stock_data_db"
        self.engine = create_engine(self.connection_string, pool_pre_ping=True, pool_recycle=3600)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables if they don't exist
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database cache tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database cache tables: {e}")
            raise


    def store(self, data: str) -> bool:
        # Logic to store data into the database
        print(f"Storing data: {data}")
        return True

database_client = DatabaseClient()

# create a tool to store data into database
@tool("StockDataStorageTool")
def store_stock_data(stock_analysis_data: StockAnalysisData) -> bool:
    """
    Tool to store researched stock data into the database.
    Use this tool only to store data into database
    :param data:
    :return:
    """
    try:
        session = database_client.SessionLocal()
        print(f"trying to save stock data: {stock_analysis_data}")
        stock_data = StockMarketAnalysisData(
            stock_name=stock_analysis_data.stock_name,
            stock_code = stock_analysis_data.stock_code,
            market=stock_analysis_data.market,
            buy_price=stock_analysis_data.buy_price,
            target_price_daily=stock_analysis_data.target_price_daily,
            target_price_weekly=stock_analysis_data.target_price_weekly,
            stop_loss=stock_analysis_data.stop_loss,
            analysis_date=stock_analysis_data.analysis_date_time
        )
        session.add(stock_data)
        session.commit()
        logger.info(f"Stored stock data for {stock_analysis_data.stock_name} successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to store stock data: {e}")
        return False
    finally:
        session.close()




