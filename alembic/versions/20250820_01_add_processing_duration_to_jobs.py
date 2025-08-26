"""
Add processing_duration to jobs

Revision ID: 20250820_01
Revises: 
Create Date: 2025-08-20
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250820_01'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('jobs') as batch_op:
        batch_op.add_column(sa.Column('processing_duration', sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table('jobs') as batch_op:
        batch_op.drop_column('processing_duration')

